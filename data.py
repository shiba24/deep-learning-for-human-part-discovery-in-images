import os
import glob
import numpy as np
import cv2
import scipy.io as sio
from tqdm import tqdm


SEED = 16
np.random.seed(SEED)

X_file_extension = ".jpg"
y_file_extension = ".mat"
parts_list = ['head', 'leye', 'reye', 'lear', 'rear',
              'lebrow', 'rebrow', 'nose', 'mouth', 'hair',
              'torso', 'neck', 'llarm', 'luarm', 'lhand',
              'rlarm', 'ruarm', 'rhand', 'llleg', 'luleg',
              'lfoot', 'rlleg', 'ruleg', 'rfoot']
merged_parts_list = {'head': 1, 'leye': 1, 'reye': 1, 'lear': 1, 'rear': 1,
                     'lebrow': 1, 'rebrow': 1, 'nose': 1, 'mouth': 1, 'hair': 1,
                     'torso': 2, 'neck': 2,
                     'llarm': 3, 'luarm': 4, 'lhand': 5,
                     'rlarm': 6, 'ruarm': 7, 'rhand': 8,
                     'llleg': 9, 'luleg': 10, 'lfoot': 11,
                     'rlleg': 12, 'ruleg': 13, 'rfoot': 14}

class MiniBatchLoader(object):
    def __init__(self, X_dir, y_dir, batchsize, insize=300, train=True):
        self.X_dir = X_dir
        self.y_dir = y_dir
        self.batchsize = batchsize
        self.insize = insize
        self.train = train
        self.train_X_file_list, self.train_y_file_list, self.test_X_file_list, self.test_y_file_list = self.split_train_test(X_dir, y_dir)
        self.human_y_list = []

    def get_file_list(self, directory, file_extension):
        if isinstance(directory, str):
            directory = [directory]
        file_list = []
        for d in directory:
            if d[-1] != os.sep:
                d += os.sep
            file_list += glob.glob(d + "*" + file_extension)
        return file_list

    def split_train_test(self, X_dir, y_dir, split_ratio=.9, y_list=None):
        all_X_list = self.get_file_list(X_dir, X_file_extension)
        all_y_list = []
        all_y = []
        if y_list is not None:
            all_y_list = y_list
        else:
            all_y_list = self.get_file_list(y_dir, y_file_extension)
        all_X = [f[f.rfind(os.sep) + 1:f.rfind(".")] for f in all_X_list]
        all_y = [f[f.rfind(os.sep) + 1:f.rfind(".")] for f in all_y_list]
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
        if self.train:
            self.random_index = np.random.permutation(self.datasize_train)
        else:
            self.random_index = np.random.permutation(self.datasize_test)

    def next(self):       # for each loop
        if self.train:
            try:
                _ = self.current_index + 1
            except AttributeError:
                print("Create Iterator settings")
                self.initialize_iterator()
            finally:
                ind_Xy = self.random_index[self.current_index:self.current_index + self.batchsize]
                # make minibatch
                minibatch_path_X = [self.train_X_file_list[ind_Xy[i]] for i in range(0, self.batchsize)]
                minibatch_path_y = [self.train_y_file_list[ind_Xy[i]] for i in range(0, self.batchsize)]
                minibatch_X, minibatch_y = self.load_batch(minibatch_path_X, minibatch_path_y)
                minibatch_X, minibatch_y = self.process_batch(minibatch_X, minibatch_y)

                self.current_index += self.batchsize
                if self.current_index + self.batchsize > self.datasize_train:
                    del self.current_index, self.random_index  # for try-catch
                    raise StopIteration
                return minibatch_X, minibatch_y
        else:
            try:
                _ = self.current_index + 1
            except AttributeError:
                print("Create Iterator settings")
                self.initialize_iterator()
            finally:
                ind_Xy = self.random_index[self.current_index:self.current_index + self.batchsize]
                # make minibatch
                minibatch_path_X = [self.test_X_file_list[ind_Xy[i]] for i in range(0, self.batchsize)]
                minibatch_path_y = [self.test_y_file_list[ind_Xy[i]] for i in range(0, self.batchsize)]
                minibatch_X, minibatch_y = self.load_batch(minibatch_path_X, minibatch_path_y)
                minibatch_X, minibatch_y = self.process_batch(minibatch_X, minibatch_y)

                self.current_index += self.batchsize
                if self.current_index + self.batchsize > self.datasize_test:
                    del self.current_index, self.random_index  # for try-catch
                    raise StopIteration
                return minibatch_X, minibatch_y

    def scan_for_human(self):
        print 'scanning all images for human part labels ... '
        all_y_list = self.train_y_file_list + self.test_y_file_list
        all_y_array = np.array([self.make_mask(f) for f in tqdm(all_y_list)])
        self.human_y_list = [all_y_list[i] for i, mat in enumerate(all_y_array) if np.any(mat > 0)]
        self.train_X_file_list, self.train_y_file_list, self.test_X_file_list, self.test_y_file_list = self.split_train_test(self.X_dir, self.y_dir, 0.9, self.human_y_list)
        print 'found %d images' % len(self.human_y_list)
        return len(self.human_y_list)

    # apply for minibatch
    def load_batch(self, minibatch_path_X, minibatch_path_y):
        minibatch_X = self.load_X(minibatch_path_X)
        minibatch_y = self.load_y(minibatch_path_y)
        return minibatch_X, minibatch_y

    def load_X(self, minibatch_path):
        return np.array([cv2.resize(cv2.imread(f), (self.insize, self.insize)) for f in minibatch_path])

    def load_y(self, minibatch_path):
        return np.array([self.make_mask(f) for f in minibatch_path])

    def make_mask(self, matfile):
        d = sio.loadmat(matfile)
        if "image" in matfile:
            parts_mask = np.transpose(np.expand_dims(d["M"], 0), (1, 2, 0))
        else:
            objects = d["anno"][0, 0][1]
            object_name = [objects[0, i][0][0] for i in range(objects.shape[1])]
            img_shape = objects[0, 0][2].shape
            parts_mask = np.zeros(img_shape + (1, ))
            for index, obj in enumerate(object_name):
                if obj == "person":
                    if not objects[0, index][3].shape == (0, 0):
                        for j in range(objects[0, index][3].shape[1]):
                            parts_mask[:, :, 0] = np.where(parts_mask[:, :, 0] == 0, merged_parts_list[objects[0, index][3][0, j][0][0]] * np.array(objects[0, index][3][0, j][1]), parts_mask[:, :, 0])
        parts_mask = cv2.resize(parts_mask.astype(np.uint8), (self.insize, self.insize), interpolation = cv2.INTER_NEAREST)
        parts_mask = (parts_mask > 0).astype(np.uint8)
        return parts_mask

    def process_batch(self, minibatch_X, minibatch_y):
        if self.train:
            change_index = np.random.random((minibatch_X.shape[0], 4))
            delta_hue = np.random.uniform(-18, 18, (minibatch_X.shape[0])).astype(np.int8)            # in opencv, hue is [0, 179]
            processed_X = np.array([self.change_shape_3d(self.change_hue(minibatch_X[i, :, :, :], delta_hue[i]),
                                                         change_index[i]) for i in range(len(minibatch_X))])
            processed_y = np.array([self.change_shape_2d(minibatch_y[i, :, :],
                                                         change_index[i]) for i in range(len(minibatch_y))])
        else:
            processed_X = minibatch_X
            processed_y = minibatch_y
        reshaped_X = np.transpose(self.standardize(processed_X), (0, 3, 1, 2))        # n_batch, n_channel, h, w        
        # reshaped_y = np.transpose(np.array([(processed_y == i + 1).astype(np.int32) for i in range(len(parts_list) + 1)]), (1, 0, 2, 3))
        # reshaped_y = np.transpose(processed_y, (1, 0, 2, 3))
        return reshaped_X, processed_y

    def standardize(self, images, mean_image="mean.jpg"):
        if not os.path.exists(mean_image):
            self.calc_mean()
        # mean = cv2.imread(mean_image)
        subtracted_img = images - 126
        return subtracted_img / 255.

    def calc_mean(self):
        pass

    # apply for each image
    def change_hue(self, img, delta_hue):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] += delta_hue
        hued_img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        return hued_img

    def change_shape_3d(self, img, change_index):
        img = self.scaling(img, change_index[0])
        img = self.rotation(img, change_index[1])
        return self.crop_3d(img, change_index[2], change_index[3])

    def change_shape_2d(self, img, change_index):
        img = self.scaling(img, change_index[0], use_integer=True)
        img = self.rotation(img, change_index[1], use_integer=True)
        return self.crop_2d(img, change_index[2], change_index[3])

    def scaling(self, img, rand_value, use_integer=False):
        scaling_factor = rand_value * 0.7 + 0.7
        resized = (int(img.shape[0] * scaling_factor), int(img.shape[1] * scaling_factor))
        if use_integer:
            resized_img = cv2.resize(img, resized, interpolation=cv2.INTER_NEAREST)
        else:
            resized_img = cv2.resize(img, resized)
        return resized_img

    def rotation(self, img, rand_value, use_integer=False):
        rotate_deg = rand_value * 60 - 30
        M = cv2.getRotationMatrix2D(img.shape[:2], rotate_deg, 1)
        if use_integer:
            rotated_img = cv2.warpAffine(img, M, img.shape[:2], flags=cv2.INTER_NEAREST)
        else:
            rotated_img = cv2.warpAffine(img, M, img.shape[:2])
        return rotated_img

    def crop_3d(self, img, rand_value1, rand_value2):
        cropped_img = np.zeros((self.insize, self.insize, img.shape[2])).astype(np.uint8)
        if img.shape[0] < self.insize or img.shape[1] < self.insize:
            x_start = np.int(rand_value1 * (self.insize - img.shape[0]))
            y_start = np.int(rand_value2 * (self.insize - img.shape[1]))
            cropped_img[x_start:x_start + img.shape[0], y_start:y_start + img.shape[1], :] = img.copy()
        elif img.shape[0] >= self.insize and img.shape[1] < self.insize:
            x_start = np.int(rand_value1 * (img.shape[0] - self.insize))
            y_start = np.int(rand_value2 * (self.insize - img.shape[1]))
            cropped_img[:, y_start:y_start + img.shape[1], :] = img[x_start:x_start + self.insize, :, :]
        elif img.shape[0] < self.insize and image.shape[1] >= self.insize:
            x_start = np.int(rand_value1 * (self.insize - img.shape[0]))
            y_start = np.int(rand_value2 * (img.shape[1] - self.insize))
            cropped_img[x_start:x_start + img.shape[0], :, :] = img[:, y_start:y_start + self.insize, :]
        elif img.shape[0] >= self.insize and img.shape[1] >= self.insize:
            x_start = np.int(rand_value1 * (img.shape[0] - self.insize))
            y_start = np.int(rand_value2 * (img.shape[1] - self.insize))
            cropped_img[:, :, :] = img[x_start:x_start + self.insize, y_start:y_start + self.insize, :]
        return cropped_img

    def crop_2d(self, img, rand_value1, rand_value2):
        cropped_img = np.zeros((self.insize, self.insize)).astype(np.uint8)
        if img.shape[0] < self.insize or img.shape[1] < self.insize:
            x_start = np.int(rand_value1 * (self.insize - img.shape[0]))
            y_start = np.int(rand_value2 * (self.insize - img.shape[1]))
            cropped_img[x_start:x_start + img.shape[0], y_start:y_start + img.shape[1]] = img.copy()
        elif img.shape[0] >= self.insize and img.shape[1] < self.insize:
            x_start = np.int(rand_value1 * (img.shape[0] - self.insize))
            y_start = np.int(rand_value2 * (self.insize - img.shape[1]))
            cropped_img[:, y_start:y_start + img.shape[1]] = img[x_start:x_start + self.insize, :]
        elif img.shape[0] < self.insize and image.shape[1] >= self.insize:
            x_start = np.int(rand_value1 * (self.insize - img.shape[0]))
            y_start = np.int(rand_value2 * (img.shape[1] - self.insize))
            cropped_img[x_start:x_start + img.shape[0], :] = img[:, y_start:y_start + self.insize]
        elif img.shape[0] >= self.insize and img.shape[1] >= self.insize:
            x_start = np.int(rand_value1 * (img.shape[0] - self.insize))
            y_start = np.int(rand_value2 * (img.shape[1] - self.insize))
            cropped_img[:, :] = img[x_start:x_start + self.insize, y_start:y_start + self.insize]
        return cropped_img

"""

import os, glob
import numpy as np
import cv2
from scipy.ndimage import zoom
import scipy.io as sio
import data

parts_list = ['head', 'leye', 'reye', 'lear', 'rear',
              'lebrow', 'rebrow', 'nose', 'mouth', 'hair',
              'torso', 'neck', 'llarm', 'luarm', 'lhand',
              'rlarm', 'ruarm', 'rhand', 'llleg', 'luleg',
              'lfoot', 'rlleg', 'ruleg', 'rfoot']
merged_parts_list = {'head':1, 'leye':1, 'reye':1, 'lear':1, 'rear':1, 'lebrow':1, 'rebrow':1, 'nose':1, 'mouth': 1, 'hair': 1, 'torso': 2, 'neck': 2, 'llarm': 3, 'luarm': 4, 'lhand':5, 'rlarm': 6, 'ruarm': 7, 'rhand': 8, 'llleg': 9, 'luleg': 10, 'lfoot':11, 'rlleg': 12, 'ruleg': 13, 'rfoot': 14}

X_dir = "./data/img/"
y_dir = "./data/mask/"

reload(data)
m = data.MiniBatchLoader(X_dir, y_dir, 20, 300)
m.initialize_iterator()
ind_Xy = m.random_index[:10]
minibatch_path_X = [m.train_X_file_list[ind_Xy[i]] for i in range(0, 10)]
minibatch_path_y = [m.train_y_file_list[ind_Xy[i]] for i in range(0, 10)]
minibatch_X, minibatch_y = m.load_batch(minibatch_path_X, minibatch_path_y)
processed_X, processed_y = m.process_batch(minibatch_X, minibatch_y)

"""
