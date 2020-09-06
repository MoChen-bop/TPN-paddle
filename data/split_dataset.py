import os
import numpy as np
import cv2
import sys
import glob
import pickle
from multiprocessing import Pool


def split_dataset(dataset_dir):
    label_dict = np.load(os.path.join(dataset_dir, 'label_dict.npy'), allow_pickle=True).item()
    print(label_dict)

    train_list_dir = os.path.join(dataset_dir, 'train.txt')
    val_list_dir = os.path.join(dataset_dir, 'val.txt')
    train_list_file = open(train_list_dir, 'w')
    val_list_file = open(val_list_dir, 'w')

    count = 1
    train_list_count = 0
    val_list_count = 0
    for label_name, label in label_dict.items():
        rgb_dir = os.path.join(dataset_dir, 'rgb', label_name)

        videos_name = os.listdir(rgb_dir)
        for video_name in videos_name:
            rgb_path = os.path.join(rgb_dir, video_name)
            _rgb_path = os.path.join('HMDB_51', 'rgb', label_name, video_name)

            frame_len = len(os.listdir(rgb_path))

            if count % 10 == 0:
                val_list_file.write(_rgb_path + ' ' + str(frame_len) + ' ' + str(label) + '\n')
                val_list_count += 1
            else:
                train_list_file.write(_rgb_path + ' ' + str(frame_len) + ' ' + str(label) + '\n')
                train_list_count += 1        
            count += 1
    
    val_list_file.close()
    train_list_file.close()

    print('train samples: ' + str(train_list_count))
    print('val samples: ' + str(val_list_count))


if __name__ == '__main__':
    split_dataset('/home/aistudio/TPN/data/HMDB_51')