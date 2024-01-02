import torch
import os
from tqdm import tqdm

VIEW = True
VIEW_TITLE = 'similarity'
WAITKEY_TIME = 0
SPLIT = False
IMG_size = 224
Gpu_id = 0
device = torch.device(f'cuda:{Gpu_id}') if Gpu_id is not None else torch.device('cpu')
SPLIT = False
MAX_NUM_TOP_IDX = 5
NUM_TOP_IDX = [1, 3, 5]
# gallery_path = "/storage/sjpark/VGGFace2/vgg_test_6s_hs_CBAM_9601"
gallery_path = "/storage/hrlee/vggface2/demo/gallery"
probe_path = "/storage/hrlee/vggface2/demo/probe"
GALLERY = dict(path=gallery_path, max_num_imgs=20, max_num_gallery_imgs=40)

network_config = dict(
    resnet = 18,
    capdimen=48,
    numpricap=512,
    predcapdimen=64,
    num_final_cap=64,
    feature_dim = 1024,
    groups =5,
    training_gpu = '4080',
    device = device
)
WEIGHT_FILE_PATH = os.path.join("checkpoints/vggface2_yolo_checkpoints",
                                f"res{network_config['resnet']}caps_primdim{network_config['capdimen']}_preddim_{network_config['predcapdimen']}_{network_config['numpricap']}_{network_config['num_final_cap']}",
                                f"RTX{network_config['training_gpu']}",
                                "top1_0_894_harmonic_0_924.pth")
# WEIGHT_FILE_PATH = "checkpoints/vggface2_checkpoints/best_res18_group5_featdim1024_top1_0_8702064896755162.pth"
network_config.setdefault('weight', WEIGHT_FILE_PATH)


if __name__=='__main__':
    from demo_core import face_similarity
    from demo_core import load_gallery_probe

    g_files, p_files = load_gallery_probe(GALLERY,probe_path, split=SPLIT)
    FACE_SIMILARITY = face_similarity(
        dict(
            imgsize=IMG_size,
            network_cfg= network_config
        ))
    FACE_SIMILARITY.make_gallery_feats_database(g_files)

    import cv2

    cv2.namedWindow(VIEW_TITLE, flags=cv2.WINDOW_NORMAL)

    cnt_correct = dict()
    for num_top in NUM_TOP_IDX:
        cnt_correct.setdefault(str(num_top), 0)
    for p_file in tqdm(p_files, total=len(p_files)):
        probe_img = torch.Tensor(cv2.imread(p_file[1])).permute(2, 0, 1)
        probe_id = p_file[0]

        top5_ids, top5_scores, resized_probe_img, top5_g_imgs = (
            FACE_SIMILARITY.find_top5_face_ids(probe_img, MAX_NUM_TOP_IDX, verbose=False))

        if VIEW:
            result_img = FACE_SIMILARITY.view_result(resized_probe_img, probe_id, top5_ids, top5_scores, top5_g_imgs)
            cv2.imshow(VIEW_TITLE, result_img)
            cv2.waitKey(WAITKEY_TIME)

        for num_top in NUM_TOP_IDX:
            cnt_correct[str(num_top)] += 1 if probe_id in top5_ids[:num_top] else 0



    print("=" * 5, "RESULTS", "=" * 5)
    print("Number of Probe Images: {}".format(GALLERY["max_num_imgs"]))
    print("Number of Gallery Images: {}".format(GALLERY["max_num_gallery_imgs"]))
    for num_top in NUM_TOP_IDX:
        print(f"top {num_top}: {cnt_correct[str(num_top)] / float(len(p_files))}")
    print("=" * 20)

    cv2.destroyAllWindows()

