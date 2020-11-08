from random import choice
import os
import cv2
import numpy as np
from utils.pipeline_helper import *
import json


class DataGenerator:
    def __init__(self, path, resize, batch_size):
        """
        Parse the data set
        categories = [name_1, name_2, ...]
        category_indexer = [ name_1:  [image_id_1, iamge_id_2, ..., image_id_n],
                        name_1:  [image_id_1, iamge_id_2, ..., image_id_n],...
                    ]
        train_labels = {
                            image_id_1: {'path': path, 'size':(width, height),
                                        'objects': [
                                                        {'category': category_name, 'bbox': [x1, y1, x2, y2]},
                                                        {'category': category_name, 'bbox': [x1, y1, x2, y2]},...
                                                   ]
                                        },...
                        }
        val_labels has the same structure with train_labels
        category_counter = {'category_name': number of trained_image, ...}
        :param path: Dataset path
        :param resize: image size for the model, [width, height, channels]
        :param batch_size: generator batch size
        """
        self.path = path
        self.resize = resize
        self.batch_size = batch_size

        self.categories = []
        self.category_indexer = {}

        self.train_labels = {}
        self.val_labels = {}

        if not self.load_dataset():
            self.parse_dataset()

        self.category_counter = {category: 0 for category in self.categories}

    def parse_dataset(self, *args, **kwargs):
        """
        Parse dataset
        :return:
        """
        pass

    def balanced_gen(self, model='yolo_v1', *args, **kwargs):
        """
        Sample balanced data generator, only used for training pipeline
        :param model: model name
        :param args: arguments passed to label parser
        :param kwargs: keyword arguments passed to label parser
        :return: generator
        """

        # parser is the function used to convert formatted data label into model ground truth matrix
        parser = self.__set_parser__(model)

        def generator():
            while True:
                indexes = self.__next_annotation_batch__()  # generate a batch of image ids
                images, labels = self.parse_batch_indexes(indexes, parser, 'train', *args, **kwargs)

                yield images, labels

        return generator()

    def sequential_gen(self, dataset='val', model='yolo_v1', endless=False, *args, **kwargs):
        parser = self.__set_parser__(model)

        def generator():
            annotations = self.__set_annotations__(dataset)
            while True:
                for indexes in annotations:
                    images, labels = self.parse_batch_indexes(indexes, parser, dataset, *args, **kwargs)
                    yield [images, labels]
                if not endless:
                    break
                else:
                    annotations = self.__set_annotations__(dataset)

        return generator()

    def parse_batch_indexes(self, indexes, parser, dataset, *args, **kwargs):
        """
        Parse a batch of image ids
        :param indexes: a list of image index, [image_id_1, image_id_2, ..., image_id_n]
        :param parser: parser function, receive *args, and **kwargs as input
        :param args:
        :param kwargs:
        :return:
        """
        images = []
        labels = []

        for index in indexes:
            if dataset == 'train':
                image_annotation = self.train_labels[index]
            else:
                image_annotation = self.val_labels[index]
            image, label = parser(image_annotation)
            images.extend(image)
            labels.extend(label)
            if len(images) > 32:
                break

        images = np.array(images)
        images = normalize_m11(images)
        labels = np.array(labels)
        return images, labels

    def __set_parser__(self, model):
        if model == 'gan':
            parser = self.gan_parser
        else:
            parser = self.gan_parser
        return parser

    def __set_annotations__(self, dataset):
        """
        split annotation batch
        :param dataset: if val, split val dataset, otherwise split training dataset
        :return:
        """
        if dataset == 'val':
            annotations = sorted(self.val_labels.keys())
        else:
            annotations = sorted(self.train_labels.keys())
        return [annotations[idx:idx + self.batch_size] for idx in range(0, len(annotations), self.batch_size)]

    def __next_annotation_batch__(self):
        """
        Generate a batch of image indexes in accordance with data balance,
        i.e. if category_i has minimum trained images, then next image should be choiced from this class.
        :return: [image_id_1, image_id_2, ..., image_id_k]
        """
        annotation_batch = []
        # check minimum category, if minimum < maximum/3, then next category is the minimum category
        # otherwise, randomly choose one
        # from category_indexer pick a image_id
        next_categories = self.__check_min__()
        for category in next_categories:
            current_indexer = choice(self.category_indexer[category])
            annotation_batch.append(current_indexer)

        return annotation_batch

    def __check_min__(self):
        """
        Check the least trained action,
        if the least is half or less than the most, then generate the least trained action
        otherwise randomly choice an action

        Return:
            action name --  action name of the next action
        """
        sorted_categories = sorted(self.category_counter.keys(), key=(lambda k: self.category_counter[k]))
        return sorted_categories[:self.batch_size]

    def __check_max__(self, category):
        """
        Check if the current category trained too many
        if the least is half or less than the most, then generate the least trained action
        otherwise randomly choice an action

        Return:
            action name --  action name of the next action
        """
        minimum_category = min(self.category_counter.keys(), key=(lambda k: self.category_counter[k]))
        cur_number = self.category_counter[category]
        if cur_number - self.category_counter[minimum_category] > 50:
            return True
        else:
            self.category_counter[category] += 1
            return False

    def gan_parser(self, image_annotation):
        images, labels = [], []

        image_path = image_annotation['path']
        objs = image_annotation['objects']

        raw_image = cv2.imread(image_path)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        for obj in objs:
            bbox = obj['bbox']
            category = obj['category']
            if category not in self.categories:
                continue
            if self.__check_max__(category) or (int(bbox[3]) - int(bbox[1])) * (int(bbox[2]) - int(bbox[0])) < 256:
                continue
            category_idx = self.categories.index(category)

            obj_image = raw_image[int(bbox[1]):int(bbox[3]), int(bbox[0]): int(bbox[2]), :]
            obj_resize = cv2.resize(obj_image, (self.resize[0], self.resize[1]))

            images.append(obj_resize)
            # label = tf.keras.utils.to_categorical(category_idx, len(self.categories))
            labels.append(category_idx)
        return images, labels

    def save_dataset(self):
        """
        Save the dataset to base path, so that it does not need to parse all the annotation each time
        :return: None
        """
        dataset_path = os.path.join(self.path, 'dataset.json')
        dataset = {
            'categories': self.categories,
            'category_indexer': self.category_indexer,
            'train_labels': self.train_labels,
            'val_labels': self.val_labels
        }
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f)
        return

    def load_dataset(self):
        """
        Load th
        :return:
        """
        dataset_path = os.path.join(self.path, 'dataset.json')
        try:
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
            self.categories = dataset['categories']
            self.category_indexer = dataset['category_indexer']
            self.train_labels = dataset['train_labels']
            self.val_labels = dataset['val_labels']
            return True
        except (KeyError, FileNotFoundError, IOError):
            return False
