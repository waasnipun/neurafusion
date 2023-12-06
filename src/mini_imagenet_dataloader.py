##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## NUS School of Computing
## Email: yaoyao.liu@nus.edu.sg
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import random
import numpy as np
from tqdm import trange
import imageio

class MiniImageNetDataLoader(object):
    def __init__(self, shot_num, way_num, episode_test_sample_num, shuffle_images=False):
        self.shot_num = shot_num
        self.way_num = way_num
        self.episode_test_sample_num = episode_test_sample_num
        self.num_samples_per_class = episode_test_sample_num + shot_num
        self.shuffle_images = shuffle_images
        miniImageNet_path = os.path.join(os.getcwd(), 'datasets/miniImageNet')
        metatrain_folder = os.path.join(miniImageNet_path, 'train')
        metaval_folder = os.path.join(miniImageNet_path, 'val')
        metatest_folder = os.path.join(miniImageNet_path, 'test')

        npy_dir = 'episode_filename_list/'
        if not os.path.exists(npy_dir):
            os.mkdir(npy_dir)

        self.npy_base_dir = npy_dir + str(self.shot_num) + 'shot_' + str(self.way_num) + 'way_' + str(
            episode_test_sample_num) + 'shuffled_' + str(self.shuffle_images) + '/'
        if not os.path.exists(self.npy_base_dir):
            os.mkdir(self.npy_base_dir)

        self.metatrain_folders = [os.path.join(metatrain_folder, label) \
                                  for label in os.listdir(metatrain_folder) \
                                  if os.path.isdir(os.path.join(metatrain_folder, label)) \
                                  ]
        self.metaval_folders = [os.path.join(metaval_folder, label) \
                                for label in os.listdir(metaval_folder) \
                                if os.path.isdir(os.path.join(metaval_folder, label)) \
                                ]
        self.metatest_folders = [os.path.join(metatest_folder, label) \
                                 for label in os.listdir(metatest_folder) \
                                 if os.path.isdir(os.path.join(metatest_folder, label)) \
                                 ]

    def get_images(self, paths, labels, nb_samples=None, shuffle=True):
        if nb_samples is not None:
            sampler = lambda x: random.sample(x, nb_samples)
        else:
            sampler = lambda x: x
        images = [(i, os.path.join(path, image)) \
                  for i, path in zip(labels, paths) \
                  for image in sampler(os.listdir(path))]
        if shuffle:
            random.shuffle(images)
        return images

    def generate_data_list(self, phase='train', episode_num=None):
        if phase == 'train':
            folders = self.metatrain_folders
            if episode_num is None:
                episode_num = 20000
            if not os.path.exists(self.npy_base_dir + '/train_filenames.npy'):
                print('Generating train filenames')
                all_filenames = []
                for _ in trange(episode_num):
                    sampled_character_folders = random.sample(folders, self.way_num)
                    random.shuffle(sampled_character_folders)
                    labels_and_images = self.get_images(sampled_character_folders, range(self.way_num),
                                                        nb_samples=self.num_samples_per_class,
                                                        shuffle=self.shuffle_images)
                    labels = [li[0] for li in labels_and_images]
                    filenames = [li[1] for li in labels_and_images]
                    all_filenames.extend(filenames)
                np.save(self.npy_base_dir + '/train_labels.npy', labels)
                np.save(self.npy_base_dir + '/train_filenames.npy', all_filenames)
                print('Train filename and label lists are saved')

        elif phase == 'val':
            folders = self.metaval_folders
            if episode_num is None:
                episode_num = 600
            if not os.path.exists(self.npy_base_dir + '/val_filenames.npy'):
                print('Generating val filenames')
                all_filenames = []
                for _ in trange(episode_num):
                    sampled_character_folders = random.sample(folders, self.way_num)
                    random.shuffle(sampled_character_folders)
                    labels_and_images = self.get_images(sampled_character_folders, range(self.way_num),
                                                        nb_samples=self.num_samples_per_class,
                                                        shuffle=self.shuffle_images)
                    labels = [li[0] for li in labels_and_images]
                    filenames = [li[1] for li in labels_and_images]
                    all_filenames.extend(filenames)
                np.save(self.npy_base_dir + '/val_labels.npy', labels)
                np.save(self.npy_base_dir + '/val_filenames.npy', all_filenames)
                print('Val filename and label lists are saved')

        elif phase == 'test':
            folders = self.metatest_folders
            if episode_num is None:
                episode_num = 600
            if not os.path.exists(self.npy_base_dir + '/test_filenames.npy'):
                print('Generating test filenames')
                all_filenames = []
                for _ in trange(episode_num):
                    sampled_character_folders = random.sample(folders, self.way_num)
                    random.shuffle(sampled_character_folders)
                    labels_and_images = self.get_images(sampled_character_folders, range(self.way_num),
                                                        nb_samples=self.num_samples_per_class,
                                                        shuffle=self.shuffle_images)
                    labels = [li[0] for li in labels_and_images]
                    filenames = [li[1] for li in labels_and_images]
                    all_filenames.extend(filenames)
                np.save(self.npy_base_dir + '/test_labels.npy', labels)
                np.save(self.npy_base_dir + '/test_filenames.npy', all_filenames)
                print('Test filename and label lists are saved')
        else:
            print('Please select vaild phase')

    def load_list(self, phase='train'):
        if phase == 'train':
            self.train_filenames = np.load(self.npy_base_dir + 'train_filenames.npy').tolist()
            self.train_labels = np.load(self.npy_base_dir + 'train_labels.npy').tolist()

        elif phase == 'val':
            self.val_filenames = np.load(self.npy_base_dir + 'val_filenames.npy').tolist()
            self.val_labels = np.load(self.npy_base_dir + 'val_labels.npy').tolist()

        elif phase == 'test':
            self.test_filenames = np.load(self.npy_base_dir + 'test_filenames.npy').tolist()
            self.test_labels = np.load(self.npy_base_dir + 'test_labels.npy').tolist()

        elif phase == 'all':
            self.train_filenames = np.load(self.npy_base_dir + 'train_filenames.npy').tolist()
            self.train_labels = np.load(self.npy_base_dir + 'train_labels.npy').tolist()

            self.val_filenames = np.load(self.npy_base_dir + 'val_filenames.npy').tolist()
            self.val_labels = np.load(self.npy_base_dir + 'val_labels.npy').tolist()

            self.test_filenames = np.load(self.npy_base_dir + 'test_filenames.npy').tolist()
            self.test_labels = np.load(self.npy_base_dir + 'test_labels.npy').tolist()

        else:
            print('Please select vaild phase')