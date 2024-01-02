def dataset_info(dataset_name='demo'):
    if dataset_name == "VGGFace2":
        train_path = "/storage/sjpark/VGGFace2/vgg_train_6s_hs_CBAM_9601"
        cache_file = "/storage/hrlee/vggface2/cache/vgg_train_6s_hs_CBAM_9601_2593_500.pickle"
        num_classes = 2593
    elif dataset_name =='total_data':
        train_path = "/storage/sjpark/total_data/train"
        cache_file = "/storage/sjpark/total_data/cache/total_train_img.pickle"
        num_classes = 79259
    elif dataset_name == 'modified_vgg':
        train_path = "/storage/sjpark/VGGFace2/vgg_train_6s_hs_SW_960"
        cache_file = "/storage/sjpark/VGGFace2/cache/vgg_train_6s_hs_SW_960.pikcle"
        num_classes = 2593

    else:
        NotImplementedError("no dataset_name!")

    gallery_path = "/storage/hrlee/groupface/demo_eval/gallery/"
    probe_path = "/storage/hrlee/groupface/demo_eval/probe/"

    test_path = "/storage/sjpark/VGGFace2/vgg_test_6s_hs_CBAM_9601"

    return dataset_name, train_path, cache_file, gallery_path, probe_path, num_classes, test_path

def get_config_dict():

    dataset_name = "modified_vgg"
    name, train_path, cache_file, gallery_path, probe_path, num_classes, test_path = dataset_info(dataset_name)
    num_class = cache_file.split('.')[:-1][0].split('_')[-2]
    img_file_per_folder = cache_file.split('.')[:-1][0].split('_')[-1]

    checkpoints_save_path = "./checkpoints/"
    checkpoints_file = None # or my checkpoint_path
    loss = 'focal_loss'

    dataset = dict(
        name = name,
        train_path= train_path,
        cache_file = cache_file,
        num_classes = num_classes,
        gallery_path = gallery_path,
        probe_path = probe_path,
        img_file_per_folder = img_file_per_folder,
        log_dir_name = f'_group_img_total_data_{img_file_per_folder}_sungjun',
        image_size = 224,
        batch_size = 16,
        num_workers = 5,
        checkpoints_file = checkpoints_file,
        checkpoints_save_path = checkpoints_save_path,
        test_path = test_path
    )
    model = dict(
        name = 'resnet',
        num = 18
    )
    solver = dict(
        epoch=100,
        lr = 1e-4,
        lr_base= 1e-5,
        lr_max = 0.5e-7,
        lr_gamma = 0.9,
        lrf = 1e-2,
        T_up = 10,
        T_down = 10,
        weight_decay = 5e-4,
        print_freq = 100,
        eval_interval = 25000 * 4,
        num_thres = 120,
        num_up_down = 50,
    )
    option = dict(
        loss=loss,
        feature_dim=1024,
        groups=5,
        gpu_id='0',
        fc_metric = 'arc',
        easy_margin=False,
        optimizer = 'adam',
        scheduler = 'cosine'
    )
    config = dict(
        dataset = dataset,
        model = model,
        solver = solver,
        option = option
    )

    return config

