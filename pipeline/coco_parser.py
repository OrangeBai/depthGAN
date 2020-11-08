from pipeline.base_parser import *


class COCOParser(DataGenerator):
    """
    CoCo data generator for Yolo_V1 model
    """

    def __init__(self, path, resize, batch_size):
        """
        Constructor
        :param resize: image size for the model, [width, height, channels]
        :param batch_size: generator batch size
        """
        super().__init__(path, resize, batch_size)
        self.super_category = self.load_super_category()
        print(1)

    def set_super_category(self, category):
        obj_indexes = [i for i, val in enumerate(self.super_category) if val == category]
        obj_names = [self.categories[i] for i in obj_indexes]
        delete_categories = [category for category in self.categories if category not in obj_names]
        for delete_category in delete_categories:
            del self.category_counter[delete_category]
            del self.category_indexer[delete_category]
            self.categories.remove(delete_category)

    def load_super_category(self):
        super_category_file = os.path.join(self.path, 'super_category.json')
        if not os.path.exists(super_category_file):
            annotation_path = os.path.join(self.path, 'annotations')
            train_path = os.path.join(annotation_path, 'instances_train2017.json')
            with open(train_path, 'r') as f:
                annotation_file = json.load(f)
            super_category = [category['supercategory'] for category in annotation_file['categories']]
            with open(super_category_file, 'w') as f:
                json.dump(super_category, f)
        else:
            with open(super_category_file, 'r') as f:
                super_category = json.load(f)
        return super_category

    def parse_dataset(self, *args, **kwargs):
        annotation_path = os.path.join(self.path, 'annotations')
        train_path = os.path.join(annotation_path, 'instances_train2017.json')
        val_path = os.path.join(annotation_path, 'instances_val2017.json')
        train_image_dir = os.path.join(self.path, 'train2017')
        val_img_dir = os.path.join(self.path, 'val2017')

        self.train_labels = self.read_annotation(train_path, train_image_dir, 'train')
        self.val_labels = self.read_annotation(val_path, val_img_dir, 'val')
        self.save_dataset()

    def read_annotation(self, annotation_path, image_dir, dataset='train'):
        with open(annotation_path, 'r') as f:
            annotation_file = json.load(f)

        if dataset == 'train':
            self.categories = [category['name'] for category in annotation_file['categories']]
            self.category_indexer = {category: [] for category in self.categories}
            self.category_counter = {category: 0 for category in self.categories}

        image_annotation = {}
        category_id_dict = {category['id']: category['name'] for category in annotation_file['categories']}
        for image in annotation_file['images']:
            image_path = os.path.join(image_dir, image['file_name'])
            if not os.path.exists(image_path):
                continue
            image_size = (image['width'], image['height'])
            image_id = image['id']
            image_annotation[image_id] = {
                'path': image_path,
                'size': image_size,
                'objects': [],
            }
        for annotation in annotation_file['annotations']:
            image_id = annotation['image_id']
            x1, y1, w, h = annotation['bbox']
            bbox = [x1, y1, x1 + w, y1 + h]
            category_id = annotation['category_id']
            category = category_id_dict[category_id]

            if dataset == 'train' and image_id in image_annotation.keys():
                self.category_indexer[category].append(str(image_id))

            if image_id in image_annotation.keys():
                image_annotation[image_id]['objects'].append(
                    {'category': category,
                     'bbox': bbox}
                )

        return image_annotation

    def save_dataset(self):
        super().save_dataset()
        dataset_path = os.path.join(self.path, 'super_category.json')
        dataset = {
            'super_category': self.super_categories
        }
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f)


# val_ann_path = r"F:\DataSet\COCO\annotations_trainval2017\annotations\instances_val2017.json"
# val_img_path = r"F:\DataSet\COCO\val2017\val2017"
# annotation, categories = read_annotation(val_ann_path)
#
# a = CocoYoloV1(categories, (600, 400), 32)
# val_gen = a.generator(annotation, val_img_path)
#
# for i in range(10):
#     d = next(val_gen)
#     img = d[0][0]
#     img = img * 255.0
#     label = d[1][0]
#     bbox = a.parse_result(label)[0]
#     for key, boxes in bbox.items():
#         for box in boxes:
#             cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0))
#     # for box, cls in bbox:
#     #     cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0))
#     cv2.imwrite(r"C:\Users\jzp0306\Desktop\1" + str(i) + ".jpg", img)
#     print(a)


# a = COCOParser(r'F:\DataSet\COCO', resize=(224, 224), batch_size=32)
#
# gen = COCOParser(r'F:\DataSet\COCO', resize=(128, 128), batch_size=16)
# gen.set_super_category('vehicle')
# a = gen.balanced_gen('gan')
#
# for i in range(10):
#     b = next(a)
#     image = np.float32(b[0][0] * 127.5 + 127.5)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(r"C:\Users\jzp0306\Desktop\Desktop" + str(i) + ".jpg", image)
