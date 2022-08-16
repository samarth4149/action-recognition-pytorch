

DATASET_CONFIG = {
    "st2stv2": {
        "num_classes": 174,
        "train_list_name": "train.txt",
        "val_list_name": "val.txt",
        "test_list_name": "test.txt",
        "filename_seperator": " ",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 3
    },
    "mini_st2stv2": {
        "num_classes": 87,
        "train_list_name": "mini_train.txt",
        "val_list_name": "mini_val.txt",
        "test_list_name": "mini_test.txt",
        "filename_seperator": " ",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 3
    },
    "kinetics400": {
        "num_classes": 400,
        "train_list_name": "train.txt",
        "val_list_name": "val.txt",
        "test_list_name": "test.txt",
        "filename_seperator": ";",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 30
    },
    "mini_kinetics400": {
        "num_classes": 200,
        "train_list_name": "mini_train.txt",
        "val_list_name": "mini_val.txt",
        "test_list_name": "mini_test.txt",
        "filename_seperator": ";",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 30
    },
    "moments": {
        "num_classes": 339,
        "train_list_name": "train.txt",
        "val_list_name": "val.txt",
        "filename_seperator": " ",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 0
    },
    "mini_moments": {
        "num_classes": 200,
        "train_list_name": "mini_train.txt",
        "val_list_name": "mini_val.txt",
        "filename_seperator": " ",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 0
    },
    "hmdb51": {
        "num_classes": 51,
        "train_list_name": "hmdb51_train_IBM_Repo.txt",
        "val_list_name": "hmdb51_test_IBM_Repo.txt",
        "test_list_name": "test.txt",
        "filename_seperator": " ",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 0
    },
    "hmdb51_cls_split_1": {
        "num_classes": 48,
        "train_list_name": "hmdb_train_cls_split_1.txt",
        "val_list_name": "hmdb_val_cls_split_1.txt",
        "test_list_name": "test.txt",
        "filename_seperator": ";;",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 0
    },
    "hmdb51_cls_split_2": {
        "num_classes": 45,
        "train_list_name": "hmdb_train_cls_split_2.txt",
        "val_list_name": "hmdb_val_cls_split_2.txt",
        "test_list_name": "test.txt",
        "filename_seperator": ";;",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 0
    },
    "hmdb51_cls_split_3": {
        "num_classes": 42,
        "train_list_name": "hmdb_train_cls_split_3.txt",
        "val_list_name": "hmdb_val_cls_split_3.txt",
        "test_list_name": "test.txt",
        "filename_seperator": ";;",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 0
    },
    "hmdb51_cls_split_4": {
        "num_classes": 39,
        "train_list_name": "hmdb_train_cls_split_4.txt",
        "val_list_name": "hmdb_val_cls_split_4.txt",
        "test_list_name": "test.txt",
        "filename_seperator": ";;",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 0
    },
    "hmdb51_cls_split_5": {
        "num_classes": 36,
        "train_list_name": "hmdb_train_cls_split_5.txt",
        "val_list_name": "hmdb_val_cls_split_5.txt",
        "test_list_name": "test.txt",
        "filename_seperator": ";;",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 0
    },
    "hmdb51_cls_split_6": {
        "num_classes": 33,
        "train_list_name": "hmdb_train_cls_split_6.txt",
        "val_list_name": "hmdb_val_cls_split_6.txt",
        "test_list_name": "test.txt",
        "filename_seperator": ";;",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 0
    },
    "hmdb51_cls_split_7": {
        "num_classes": 30,
        "train_list_name": "hmdb_train_cls_split_7.txt",
        "val_list_name": "hmdb_val_cls_split_7.txt",
        "test_list_name": "test.txt",
        "filename_seperator": ";;",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 0
    },
    "hmdb51_cls_split_8": {
        "num_classes": 27,
        "train_list_name": "hmdb_train_cls_split_8.txt",
        "val_list_name": "hmdb_val_cls_split_8.txt",
        "test_list_name": "test.txt",
        "filename_seperator": ";;",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 0
    },
    "hmdb51_cls_split_9": {
        "num_classes": 24,
        "train_list_name": "hmdb_train_cls_split_9.txt",
        "val_list_name": "hmdb_val_cls_split_9.txt",
        "test_list_name": "test.txt",
        "filename_seperator": ";;",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 0
    },
    "hmdb51_cls_split_10": {
        "num_classes": 21,
        "train_list_name": "hmdb_train_cls_split_10.txt",
        "val_list_name": "hmdb_val_cls_split_10.txt",
        "test_list_name": "test.txt",
        "filename_seperator": ";;",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 0
    },
    "ucf101": {
        "num_classes": 101,
        "train_list_name": "train_random_frames.txt",
        "val_list_name": "val_random_frames.txt",
        "test_list_name": "test.txt",
        "filename_seperator": " ",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 3
    },
    "diving48": {
        "num_classes": 48,
        "train_list_name": "train_random_frames.txt",
        "val_list_name": "val_random_frames.txt",
        "test_list_name": "test.txt",
        "filename_seperator": " ",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 0
    },
    "ikea_furniture": {
        "num_classes": 12,
        "train_list_name": "train_random_frames.txt",
        "val_list_name": "val_random_frames.txt",
        "test_list_name": "test.txt",
        "filename_seperator": " ",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 0
    },
    "uav_human": {
        "num_classes": 155,
        "train_list_name": "train_random_frames.txt",
        "val_list_name": "val_random_frames.txt",
        "test_list_name": "test.txt",
        "filename_seperator": ";",
        "image_tmpl": "{:05d}.jpg",
        "filter_video": 0
    }
}


def get_dataset_config(dataset):
    ret = DATASET_CONFIG[dataset]
    num_classes = ret['num_classes']
    train_list_name = ret['train_list_name']
    val_list_name = ret['val_list_name']
    test_list_name = ret.get('test_list_name', None)
    filename_seperator = ret['filename_seperator']
    image_tmpl = ret['image_tmpl']
    filter_video = ret.get('filter_video', 0)
    label_file = ret.get('label_file', None)

    return num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, \
           image_tmpl, filter_video, label_file
