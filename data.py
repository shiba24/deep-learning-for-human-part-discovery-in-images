#!/usr/bin/env python2.7
# coding:utf-8

import os, glob
import numpy as np
import cv2

SEED = 16
np.random.seed(SEED)

X_dir = "data/img/"
y_dir = "data/mask/"
X_file_extension = ".jpg"
y_file_extension = ".mat"


class TwoRandomIterator(object):
    def __iter__(self):   # iterator setting
        return self

    def next(self):       # for each loop
        return np.random.random(), np.random.random()


class MiniBatchLoader(object):
    def __init__(self, X_dir, y_dir, batchsize, insize):
        # self.processes = (self.scaling, self.rotation, self.change_hue)
        self.X_dir = X_dir
        self.y_dir = y_dir
        self.batchsize = batchsize
        self.insize = insize
        # self.R = TwoRandomIterator()
        self.train_X_file_list, self.train_y_file_list, self.test_X_file_list, self.test_y_file_list = self.split_train_test(X_dir, y_dir)

    def get_file_list(self, directory, file_extension):
        if isinstance(directory, str):
            directory = [directory]
        file_list = []
        for d in directory:
            if d[-1] != '/':
                d += '/'
            file_list += glob.glob(d + "*" + file_extension)
        return file_list

    def split_train_test(self, X_dir, y_dir, split_ratio=0.95):
        all_X_list = self.get_file_list(X_dir, X_file_extension)
        all_y_list = self.get_file_list(y_dir, y_file_extension)

        all_X = [f[f.rfind("/") + 1:f.rfind(".")] for f in all_X_list]
        all_y = [f[f.rfind("/") + 1:f.rfind(".")] for f in all_y_list]
        matched_list = [element for element in all_y if element in all_X]
        self.datasize = len(matched_list)
        self.datasize_train = int(self.datasize * split_ratio)
        self.datasize_test = self.datasize - self.datasize_train
        print("training datasets: ", self.datasize_train, "test datasets: ", self.datasize_test)

        indices = np.random.permutation(self.datasize)
        train_list = [matched_list[indices[i]] for i in range(0, self.datasize_train)]
        test_list = [matched_list[indices[i]] for i in range(self.datasize_train, self.datasize)]
        train_X_file_list = [X_dir + f + X_file_extension for f in train_list]
        train_y_file_list = [y_dir + f + y_file_extension for f in train_list]
        test_X_file_list = [X_dir + f + X_file_extension for f in test_list]
        test_y_file_list = [y_dir + f + y_file_extension for f in test_list]
        return train_X_file_list, train_y_file_list, test_X_file_list, test_y_file_list

    def __iter__(self):   # iterator setting
        return self

    # initialize for each training loop
    def initialize_iterator(self):
        self.current_index = 0
        self.random_index = np.random.permutation(self.datasize_train)
        self.process_list = np.random.randint(0, len(self.processes), self.datasize_train)

    def next(self):       # for each loop
        try:
            print(self.current_index)
        except AttributeError:
            print("Create Iterator settings")
            self.initialize_iterator()
        finally:
            ind_Xy = self.random_index[self.current_index:self.current_index + self.batchsize]
            ind_process = self.process_list[self.current_index:self.current_index + self.batchsize]
            # make minibatch
            minibatch_path_X = [self.train_X_file_list[ind_Xy[i]] for i in range(0, self.batchsize)]
            minibatch_path_y = [self.train_y_file_list[ind_Xy[i]] for i in range(0, self.batchsize)]
            minibatch_X, minibatch_y = self.load_batch(minibatch_path_X, minibatch_path_y)
            # randomly apply augmentation process
            processed_X, processed_y = self.process_batch(minibatch_X, minibatch_y)

            self.current_index += self.batchsize
            if self.current_index + self.batchsize > self.datasize:
                del self.current_index  # for try-catch
                raise StopIteration
            return processed_X, processed_y


    # apply for minibatch
    def process_batch(self, minibatch_X, minibatch_y):
        processed_X = np.array([self.change_hue(X, ind_process[index]) for index, X in enumerate(minibatch_X)])
        processed_y = np.array([self.process_y(y, ind_process[index], r) for index, y in enumerate(minibatch_y)])


    def load_batch(self, minibatch_path_X, minibatch_path_y):
        minibatch_X = self.subtract_mean_batch(self.load_X(minibatch_path_X))
        minibatch_y = self.load_y(minibatch_path_y)
        return minibatch_X, minibatch_y


    def load_X(self, minibatch_path):
        return np.array([cv2.imread(f) for f in minibatch_path])

    def load_y(self, minibatch_path):


    def subtract_mean_batch(self, images, mean_image="mean.jpg"):
        subtracted_img = images - 126
        return subtracted_img



    def change_hue(self, img):
        delta_hue = np.uint8(np.random.uniform(-18, 18))        # in opencv, hue is [0, 179]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] += delta_hue
        hued_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return hued_img

    def change_shape_Xy(self, minibatch_X, minibatch_y):
        shaped_X = np.zeros((self.batchsize, self.insize, self.insize, 3))

###???####
        shaped_y = np.zeros((self.batchsize, self.insize, self.insize, 3))
        for i in range(self.batchsize):
            r1, r2 = np.random.random(), np.random.random()
            shaped_X[i, :, :, :] = self.change_shape_one(minibatch_X[i, :, :, :], r1, r2)
####????
            shaped_y[i, :, :, :] = self.change_shape_one(minibatch_y[i, :, :, :], r1, r2)

        return shaped_X, shaped_y

    def change_shape_one(self, img, r1, r2):
        img = self.scaling(img, r1)
        img = self.rotation(img, r1)
        return self.crop(img, r1, r2) / 255.0

    # apply for each image
    def scaling(self, img, rand_value):
        scaling_factor = rand_value * 0.7 + 0.7
        resized = (int(img.shape[0] * scaling_factor), int(img.shape[1] * scaling_factor))
        resized_img = cv2.resize(img, resized)
        return resized_img

    def rotation(self, img, rand_value):
        rotate_deg = rand_value * 60 - 30
        M = cv2.getRotationMatrix2D(img.shape[:2], rotate_deg, 1)
        rotated_img = cv2.warpAffine(img, M, img.shape[:2])
        return rotated_img

    def crop(self, img, rand_value1, rand_value2):
        print img.shape, self.insize
        x_start = np.int(rand_value1 * (img.shape[0] - self.insize))
        y_start = np.int(rand_value2 * (img.shape[1] - self.insize))
        if x_start < 0 or y_start < 0:
            # raise IndexError("Input image", img.shape[:2], "is smaller than insize of the network!")
            print("Input image", img.shape[:2], "is smaller than insize of the network!")
            return self.pad(img, rand_value1, rand_value2)
        else:
            cropped_img = img[x_start:x_start + self.insize, y_start:y_start + self.insize, :]
            return cropped_img

    def pad(self, img, rand_value1, rand_value2):
        padded_img = np.ones((self.insize, self.insize, 3)).astype(np.uint8) * 177
        x_start = np.int(rand_value1 * (self.insize - img.shape[0]))
        y_start = np.int(rand_value2 * (self.insize - img.shape[1]))
        padded_img[x_start:x_start + img.shape[0], y_start:y_start + img.shape[1], :] = img.copy()
        return padded_img


    def subtract_mean_one(self, img, mean_image="mean.jpg"):
        mean_img = cv2.imread(mean_image)
        subtracted_img = img - mean_img
        return subtracted_img





