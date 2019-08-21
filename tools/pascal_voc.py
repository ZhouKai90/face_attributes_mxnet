from __future__ import print_function, absolute_import
import os
import numpy as np
from imdb import Imdb
import xml.etree.ElementTree as ET
import cv2


class PascalVoc(Imdb):
    """
    Implementation of Imdb for Pascal VOC datasets

    Parameters:
    ----------
    image_set : str
        set to be used, can be train, val, trainval, test
    devkit_path : str
        devkit path of VOC dataset
    shuffle : boolean
        whether to initial shuffle the image list
    is_train : boolean
        if true, will load annotations
    """
    def __init__(self, image_set, devkit_path, shuffle=False, is_train=False, class_names=None, names='pascal_voc.names'):
        super(PascalVoc, self).__init__('voc_' + image_set)
        self.image_set = image_set
        self.devkit_path = devkit_path
        self.data_path = os.path.join(devkit_path, 'VOC')
        self.extension = '.jpg'
        self.is_train = is_train

        if class_names is not None:
            self.classes = class_names.strip().split(',')
        else:
            self.classes = self._load_class_names(names,
                os.path.join(os.path.dirname(__file__), 'names'))

        self.config = {'use_difficult': True,
                       'comp_id': 'comp4',}

        self.num_classes = len(self.classes)
        self.image_set_index = self._load_image_set_index(shuffle)
        self.num_images = len(self.image_set_index)
        print(self.num_images)
        if self.is_train:
            self.labels = self._load_image_labels()

    @property
    def cache_path(self):
        """
        make a directory to store all caches

        Returns:
        ---------
            cache path
        """
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def _load_image_set_index(self, shuffle):
        """
        find out which indexes correspond to given image set (train or val)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        if shuffle:
            np.random.shuffle(image_set_index)
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.data_path, 'JPEGImages', name + self.extension)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index]

    def _label_path_from_index(self, index):
        """
        given image index, find out annotation path

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        label_file = os.path.join(self.data_path, 'Annotations', index + '.xml')
        assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)
        return label_file

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 5] tensor
        """
        temp = []

        # load ground-truth from xml annotations
        for idx in self.image_set_index:
            label_file = self._label_path_from_index(idx)
            tree = ET.parse(label_file)
            root = tree.getroot()
            # size = root.find('size')
            # width = float(size.find('width').text)
            # height = float(size.find('height').text)
            label = []

            for obj in root.iter('object'):
                # cls_name = obj.find('name').text
                # if cls_name not in self.classes:
                #     cls_id = len(self.classes)
                # else:
                #     cls_id = self.classes.index(cls_name)
                gender = str(obj.find('gender').text)
                mask = str(obj.find('mask').text)
                mouth = str(obj.find('mouth').text)
                eyeglass = str(obj.find('eyeglass').text)
                sunglass = str(obj.find('sunglass').text)
                blurriness = str(obj.find('blurriness').text)
                illumination = str(obj.find('illumination').text)
                print(gender, mask, mouth, eyeglass, sunglass, blurriness, illumination)
                label.append([gender, mask, mouth, eyeglass, sunglass, blurriness, illumination])
            temp.append(np.array(label))
        return temp

    def _get_imsize(self, im_name):
        """
        get image size info
        Returns:
        ----------
        tuple of (height, width)
        """
        img = cv2.imread(im_name)
        return (img.shape[0], img.shape[1])
