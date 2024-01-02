import sys, os
import torch, cv2
import numpy as np
import time
from tqdm import tqdm

from model.GroupFace_caps import GroupFace_caps
from loss.loss import ArcMarginProduct
from loss.focal_loss import FocalLoss

from dataset.dataset import VGGdataset, totaldata, torch_loader
from Config.config import get_config_dict

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import torch.nn.functional as F

from utilss.ema_single import ModelEMA
from utilss.ema_arc import ModelEMA_arc
from utilss.general import increment_name
from copy import deepcopy
import math
import random

NCOLS = 100


class Trainer():
    def __init__(self, cfg, device=torch.device('cpu')):
        self.cfg = cfg
        self.device = self.setup_device()
        self.model = self.setup_network(self.device).to(self.device)
        self.ema_model = ModelEMA(self.model)
        self.fc_metric = self.setup_fc_metric().to(self.device)
        self.ema_fc_metric = ModelEMA_arc(self.fc_metric)
        self.criterion = self.setup_loss(self.device)
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        self.global_step = 0
        self.best_top1_acc = 0.0
        self.train_loader = self.get_dataloader()
        self.writer = SummaryWriter(log_dir=self.get_log_dir())
        self.save_path = self.get_save_path()

    def pad_to_square(self,image, pad_value=0):
        _, h, w = image.shape

        difference = abs(h - w)

        # (top, bottom) padding or (left, right) padding
        if h <= w:
            top = difference // 2
            bottom = difference - difference // 2
            pad = [0, 0, top, bottom]
        else:
            left = difference // 2
            right = difference - difference // 2
            pad = [left, right, 0, 0]

        # Add padding
        image = F.pad(image, pad, mode='constant', value=pad_value)
        return image, pad

    def resize(self, image, size):
        return F.interpolate(image.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)

    def setup_device(self):
        if len(self.cfg['option']['gpu_id']) != 0:
            device = torch.device("cuda:{}".format(self.cfg['option']['gpu_id']) if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        return device

    def get_dataloader(self):
        if self.cfg['dataset']['name'] == 'VGGFace2':
            from dataset.dataset import VGGdataset
            dataset = VGGdataset(self.cfg['dataset']['image_size'], self.cfg['dataset']['train_path'], self.cfg['dataset']['cache_file'])
        elif self.cfg['dataset']['name'] == 'total_data':
            from dataset.dataset import totaldata
            dataset = totaldata(self.cfg['dataset']['image_size'], self.cfg['dataset']['train_path'], self.cfg['dataset']['cache_file'])
        else:
            raise ValueError('Invalid dataset name,'
                             'currently supported...')

        loader = torch.utils.data.DataLoader(dataset, batch_size= self.cfg['dataset']['batch_size'], shuffle = True, num_workers = self.cfg['dataset']['num_workers'])

        return loader

    def setup_network(self, device):
        model = GroupFace_caps(data_h=self.cfg['dataset']['image_size'], data_w=self.cfg['dataset']['image_size'],
                               capdimen=48, predcapdimen=64, numpricap=512,
                               num_final_cap=64,
                               feature_dim=self.cfg['option']['feature_dim'], groups=self.cfg['option']['groups']).to(device)
        return model

    def setup_loss(self, device):
        if self.cfg['option']['loss'] == 'focal_loss':
            loss = FocalLoss(gamma=2)
        else:
            loss = torch.nn.CrossEntropyLoss()
        return loss.to(device)


    def setup_fc_metric(self):
        if  self.cfg['option']['fc_metric'] == 'arc':
            fc_metric = ArcMarginProduct(in_features=self.cfg['option']['feature_dim'], out_features=self.cfg['dataset']['num_classes'], s=30, m=0.5,
                                         device=self.device, easy_margin=self.cfg['option']['easy_margin'])
        else:
            raise NotImplementedError("[setup_fc_metric] NOT IMPLEMENTED..")
        return fc_metric.to(self.device)

    def setup_optimizer(self):
        if self.cfg['option']['optimizer'] == 'adam':
            optimizer = torch.optim.Adam([{'params': self.model.parameters()}, {'params': self.fc_metric.parameters()}], lr=self.cfg['solver']['lr'],  weight_decay = self.cfg['solver']['weight_decay'])
        else:
            raise NotImplementedError("Not Implemented {}".format(self.cfg['option']['optimizer']))

        return optimizer

    def setup_scheduler(self):
        if self.cfg['option']['scheduler'] == 'cosine':
            lf = lambda x: ((1 - math.cos(x * math.pi / self.cfg['solver']['epoch'])) / 2) * (self.cfg['solver']['lrf'] - 1) + 1
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        elif self.cfg['option']['scheduler'] == 'constant':
            lf = lambda x: 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        elif self.cfg['option']['scheduler'] == 'cyclelr':
            scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.cfg['solver']['lr_base'], max_lr=self.cfg['solver']['lr_max'],
                                                          step_size_up=self.cfg['solver']['T_up'],
                                                          step_size_down=self.cfg['solver']['T_down'],
                                                          gamma=self.cfg['solver']['lr_gamma'], cycle_momentum=False,
                                                          mode='triangular2')
        elif self.cfg['option']['scheduler'] == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.cfg['solver']['T_down'], gamma=0.9)
        else:
            NotImplementedError('Not Implemented...{}'.format(self.cfg['option']['scheduler']))
        return scheduler
    def get_log_dir(self):
        log_path = str(increment_name('logging/' + self.cfg['dataset']['log_dir_name']))
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        return log_path

    def get_save_path(self):
        save_path = str(increment_name(os.path.join(self.cfg['dataset']['checkpoints_save_path'] + self.cfg['dataset']['name'].lower())))
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        return save_path
    def load_gallery(self):
        gallery_path = self.cfg['dataset']['gallery_path']
        #
        gallery_files = []
        gallery_ids = []
        gallery_len = len(os.listdir(gallery_path))
        #
        for i, dir in enumerate(os.listdir(gallery_path)):
            for file in os.listdir(os.path.join(gallery_path, dir)):
                file_path = os.path.join(gallery_path, dir, file)
                gallery_files.append(file_path)
                gallery_ids.append(dir)
            sys.stdout.write("\r>> LoadGallery[{}/{}] ".format(i, gallery_len))
            sys.stdout.flush()
        print('\n')
        return gallery_files, gallery_ids

    def load_probe(self):
        probe_path = self.cfg['dataset']['probe_path']
        probe_files = []
        probe_ids = []

        probe_len = len(os.listdir(probe_path))
        for i, dir in enumerate(os.listdir(probe_path)):
            for file in os.listdir(os.path.join(probe_path, dir)):
                file_path = os.path.join(probe_path, dir, file)
                probe_files.append(file_path)
                probe_ids.append(dir)
            sys.stdout.write("\r>> LoadProbe[{}/{}] ".format(i, probe_len))
            sys.stdout.flush()
        print('\n')
        return probe_files, probe_ids

    def load_gallery_probe(self, parameters):
        gallery_path = parameters['options'].test_path
        #
        all_path = []
        p_files = []
        p_ids = []
        g_files = []
        g_ids = []

        # make gallery_probe slicing index
        num_thres = parameters['options'].num_thres
        for i, dir in enumerate(os.listdir(gallery_path)):
            num = 0
            files_in_dir = os.listdir(os.path.join(gallery_path, dir))
            random.shuffle(files_in_dir)
            for file in files_in_dir:
                num += 1
                if num <= num_thres:
                    file_path = os.path.join(gallery_path, dir, file)
                    all_path.append(file_path)
        # split gallery_probe
        for j in range(0, len(os.listdir(gallery_path))):
            tmp = random.choice(all_path[j * num_thres: (1 + j) * num_thres])
            p_files.append(tmp)
            p_ids.append(tmp.split("/")[-2])
            for i in all_path[j * num_thres: (1 + j) * num_thres]:
                if i == tmp:
                    pass
                else:
                    g_files.append(i)
                    g_ids.append(i.split("/")[-2])

        return g_files, g_ids, p_files, p_ids

    def load_weight(self, model, ema_model, fc_metric, ema_fc_metric, optimizer, best_top1_acc, device):
        if self.cfg['dataset']['checkpoint_file'] is not None:
            file_path = os.path.join(self.cfg['dataset']['checkpoints_save_path'], self.cfg['dataset']['checkpoint_file'])
            assert os.path.exists(file_path), f'There is no weight file!!'

            print("Loading saved weights {}".format(self.cfg['dataset']['checkpoint_file']))
            ckpt = torch.load(file_path, map_location=self.device)
            #
            resume_state_dict = ckpt['model'].state_dict()  # checkpoint state_dict as FP32
            model.load_state_dict(resume_state_dict, strict=True)  # load
            #
            ema_model.ema.load_state_dict(ckpt['ema_model'].state_dict(), strict=True)
            #
            fc_metric.load_weights(ckpt['fc_metric'])
            #
            ema_fc_metric.load_weights(ckpt['ema_fc_metric'])
            #
            # optimizer.load_state_dict(ckpt['optimizer'])
            #
            # best_top1_acc = ckpt['best_top1_ac']

        return model, ema_model, fc_metric, ema_fc_metric, optimizer, best_top1_acc

    def save_model(self, ema_model, fc_metric, ema_fc_metric, save_path, optimizer, best_top1_ac=0.0, best_top5_acc=0.0, best_harmonic_acc=0.0, prefix='last'):
        save_file = '{}res{}_group{}_featdim{}_top1_{}_harmonic_{}.pth'.format(prefix, self.cfg['model']['num'], self.cfg['option']['groups'],
                                                                               self.cfg['option']['feature_dim'],
                                                                               str(round(best_top1_ac, 3)).replace('.',
                                                                                                                   '_'),
                                                                               str(round(best_harmonic_acc, 3)).replace(
                                                                                   '.', '_'),
                                                                               self.cfg['dataset']['image_size'])
        path = os.path.join(save_path, save_file)
        torch.save({'model': deepcopy(self.model),
                    'ema_model': deepcopy(ema_model.ema),
                    'fc_metric': {
                        'weight': fc_metric.weight,
                        'in_features': fc_metric.in_features,
                        'out_features': fc_metric.out_features,
                        'm': fc_metric.m,
                        's': fc_metric.s,
                        'easy_margin': fc_metric.easy_margin,
                    },
                    'best_top1_ac': best_top1_ac,
                    'best_top5_acc': best_top5_acc,
                    'best_harmonic_acc': best_harmonic_acc,
                    'ema_fc_metric': deepcopy(ema_fc_metric),
                    'optimizer': optimizer.state_dict()}, path)


    def training(self):
        device = self.device
        scheduler = self.scheduler
        model = self.model
        ema_model = self.ema_model
        fc_metric = self.fc_metric
        ema_fc_metric = self.ema_fc_metric
        optimizer = self.optimizer
        train_loader = self.train_loader
        criterion = self.criterion
        writer = self.writer
        save_path = self.save_path
        accumulate = max(1, round(64/ self.cfg['dataset']['batch_size']))
        last_opt_step = 0
        best_top1_acc = self.best_top1_acc
        best_top5_acc = 0
        best_harmonic_mean = 0.0

        # test checkpoints
        t_parameter = dict()

        t_parameter['device'] = device
        t_parameter['model'] = model.eval()
        t_parameter['writer'] = writer
        t_parameter['global_step'] = 0
        t_parameter['ema_model'] = ema_model
        t_parameter['best_eval'] = 0.0

        eval(t_parameter)

        for i in range(self.cfg['solver']['epoch']):
            model.train()
            loss_sum = 0.0
            cnt_right = 0.0
            cnt_total = 0.0
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, ascii=True, ncols=NCOLS)
            for curr_step, data in pbar:
                self.global_step = (i * len(train_loader) + (curr_step + 1))
                t_global_step = (i * len(train_loader) + (curr_step + 1))
                #
                img, file_path, id, label = data
                img = img.to(device)
                label = label.to(device).long()

                group_inter, final, group_prob, group_label = model(img)
                output = fc_metric(final, label)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                if self.global_step - last_opt_step >= accumulate:
                    optimizer.step()
                    if ema_model:
                        ema_model.update(model)
                        ema_fc_metric.update(fc_metric)
                    last_opt_step = self.global_step

                # Accumulate loss
                loss_sum += float(loss)

                #for Accuracy
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis= 1)
                label = label.data.cpu().numpy()
                cnt_right += np.sum((output == label).astype(int))
                cnt_total += label.shape[0]

                #print log data
                if self.global_step % self.cfg['solver']['print_freq'] ==0:
                    writer.add_scalar(tag='train/loss', scalar_value=loss, global_step= self.global_step)
                if self.global_step % self.cfg['solver']['eval_interval'] == 0:
                    curr_top1_acc, curr_top5_acc = eval()
                    model.train()
                    #
                    writer.add_scalar(tag='train/classification_acc', scalar_value=cnt_right / cnt_total, global_step=self.global_step)
                    pbar.set_postfix(loss=loss.detach().cpu().numpy().item(), acc=(cnt_right/cnt_total))

                    curr_hamonic_mean = 2 * (curr_top1_acc * curr_top5_acc) / (curr_top1_acc + curr_top5_acc)
                    if best_harmonic_mean < curr_hamonic_mean:
                        best_harmonic_mean = curr_hamonic_mean
                        best_top1_acc = curr_top1_acc
                        best_top5_acc = curr_top5_acc
                        self.save_model(ema_model, fc_metric, ema_fc_metric, save_path, optimizer, best_top1_acc, best_top5_acc, best_harmonic_mean, 'best_')
                        print('\n [save] top1_acc: {}, top5_acc: {}', curr_top1_acc, curr_top5_acc)

            scheduler.step()
            self.save_model(ema_model, fc_metric, ema_fc_metric, save_path, optimizer, best_top1_acc, best_top5_acc,
                            best_harmonic_mean, 'last_')
            writer.add_scalar(tag='train/lr', scalar_value= scheduler.get_lr()[0], global_step=i)
            print("\nEpoch {} Trained Loss_Sum:{:.5f}\n".format(i, loss_sum / float(len(train_loader))))


    def eval(self,parameters):
        writer = parameters['writer']
        device = parameters['device']
        model = parameters['ema_model'].ema
        model.eval().to(device)

        print('\n[eval] Get gallery and probe images...')

        gallery_files, gallery_ids, probe_files, probe_ids = self.load_gallery_probe(parameters)
        g_id_uniqe = np.unique(gallery_ids)

        set_g_index = list(range(len(gallery_files)))
        split_index = []
        tmp = []
        for i in set_g_index:
            if i != 0 and i % 32 ==0:
                split_index.append(tmp)
                tmp = []
            tmp.append(i)
            if i == set_g_index[-1]:
                split_index.append(tmp)

        gallery_feats = []
        print('\n[eval] Obtain feature vectors of gallery images...')
        for idxes_batch in tqdm(split_index, total= len(split_index)):
            batch = []
            for i in idxes_batch:
                gallery_file, _ = self.pad_to_square(torch.Tensor(cv2.imread(gallery_files[i])))
                gallery_file = self.resize(gallery_file, self.cfg['dataset']['image_size'])
                gallery_file = np.array(gallery_file.permute(1, 2, 0)).astype(np.uint8)
                batch.append(torch_loader(gallery_file, dimension=self.cfg['dataset']['image_size']).unsqueeze(0).to(device))
            batch = torch.cat(batch, dim=0)
            _, final, _, _ = model(batch)
            gallery_feat = final / torch.norm(final, p=2, dim=1, keepdim= True)
            gallery_feat = gallery_feat.detach().cpu().numpy()
            gallery_feats.append(gallery_feat)
        gallery_feats = np.concatenate(gallery_feats, axis = 0)
        #
        evaled_cnt = 0.0
        top1_cnt = dict()
        top3_cnt = dict()
        top5_cnt = dict()
        top1_mean_cnt = dict()
        top3_mean_cnt = dict()
        top5_mean_cnt = dict()
        #
        gallery_ids_np = np.expand_dims(np.array(gallery_ids), axis=0)
        num_gallery_imgs = np.linspace(10, 120, 12, dtype=np.int16)
        for num_imgs in num_gallery_imgs:
            top1_cnt[num_imgs] = 0.0
            top3_cnt[num_imgs] = 0.0
            top5_cnt[num_imgs] = 0.0
            top1_mean_cnt[num_imgs] = 0.0
            top3_mean_cnt[num_imgs] = 0.0
            top5_mean_cnt[num_imgs] = 0.0
        #
        print('\n[eval] Calculate accuracy...')
        pbar = tqdm(zip(probe_files, probe_ids), total=len(probe_ids), leave=True, mininterval=0.1, ascii=True, ncols=NCOLS)
        for probe_file, GT_id in pbar:
            evaled_cnt += 1
            probe_file, pad = self.pad_to_square(torch.Tensor(cv2.imread(probe_file)).permute(2, 0, 1))
            probe_file = self.resize(probe_file, self.cfg['dataset']['image_size'])
            probe_file = np.array(probe_file.permute(1, 2, 0)).astype(np.uint8)
            _, final, _, _ = model(torch_loader(probe_file, dimension=self.cfg['dataset']['image_size'])).unsqueeze(0).to(device)
            probe_feat = final / torch.norm(final, p=2, keepdim=False)
            probe_feat = probe_feat.detach().cpu().reshape(1, self.cfg['option']['feature_dim']).numpy()

            scores = cosine_similarity(probe_feat, gallery_feat)

            for num_imgs in num_gallery_imgs:
                selected_idxes = []
                selected_g_ids = []
                mean_score_per_g_id = []
                mean_g_id = []
                for g_id in g_id_uniqe:
                    idx_g_id = np.where(gallery_ids_np[0] == g_id)[0]
                    try:
                        idxes_ = random.sample(list(idx_g_id), num_imgs)
                        actual_num_imgs = len(idxes_)
                    except:
                        idxes_ = list(idx_g_id)
                        actual_num_imgs = len(idxes_)

                    selected_idxes.append(idxes_)
                    selected_g_ids.append([g_id] * actual_num_imgs)
                    mean_score_per_g_id.append(np.average(scores[0, idxes_]))
                    mean_g_id.append(g_id)

                selected_idxes = list(np.concatenate(selected_idxes, axis = 0))
                selected_g_ids = list(np.concatenate(selected_g_ids, axis=0))
                selected_scores = scores[0, selected_idxes]

                sort_scores_idx = np.argsort(selected_scores)
                sorted_gallery_ids = np.array(selected_g_ids)[sort_scores_idx]

                top1_id = [sorted_gallery_ids[-1]]
                top3_ids = list(sorted_gallery_ids[-3:])
                top5_ids = list(sorted_gallery_ids[-5:])
                top1_cnt[num_imgs] += 1 if GT_id in top1_id else 0
                top3_cnt[num_imgs] += 1 if GT_id in top1_id else 0
                top5_cnt[num_imgs] += 1 if GT_id in top1_id else 0

                sort_mean_scores_idx = np.argsort(mean_score_per_g_id)
                sort_mean_g_id = np.array(mean_g_id)[sort_mean_scores_idx]

                top1_id = [sort_mean_g_id[-1]]
                top3_ids = list(sort_mean_g_id[-3:])
                top5_ids = list(sort_mean_g_id[-5:])
                top1_mean_cnt[num_imgs] += 1 if GT_id in top1_id else 0
                top3_mean_cnt[num_imgs] += 1 if GT_id in top1_id else 0
                top5_mean_cnt[num_imgs] += 1 if GT_id in top1_id else 0

        for num_imgs in num_gallery_imgs:
            writer.add_scalar(tag='eval/top1_acc/{}'.format(num_imgs),
                              scalar_value=top1_cnt[num_imgs] / evaled_cnt,
                              global_step=parameters['global_step'])
            writer.add_scalar(tag='eval/top3_acc/{}'.format(num_imgs),
                              scalar_value=top3_cnt[num_imgs] / evaled_cnt,
                              global_step=parameters['global_step'])
            writer.add_scalar(tag='eval/top5_acc/{}'.format(num_imgs),
                              scalar_value=top5_cnt[num_imgs] / evaled_cnt,
                              global_step=parameters['global_step'])
            writer.add_scalar(tag='eval/top1_acc (mean)/{}'.format(num_imgs),
                              scalar_value=top1_mean_cnt[num_imgs] / evaled_cnt,
                              global_step=parameters['global_step'])
            writer.add_scalar(tag='eval/top3_acc (mean)/{}'.format(num_imgs),
                              scalar_value=top3_mean_cnt[num_imgs] / evaled_cnt,
                              global_step=parameters['global_step'])
            writer.add_scalar(tag='eval/top5_acc (mean)/{}'.format(num_imgs),
                              scalar_value=top5_mean_cnt[num_imgs] / evaled_cnt,
                              global_step=parameters['global_step'])
        return top1_cnt[120] / evaled_cnt, top5_cnt[120] / evaled_cnt
















