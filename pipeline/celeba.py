import numpy as np
from config import *
from random import choice
import cv2
import tensorflow as tf

image_names = os.listdir(celeba_dir)


def celeba_gen(batch_size, crop_size=120, resize=(64, 64)):
    batch_names = [image_names[idx:idx + batch_size] for idx in range(0, len(image_names), batch_size)]
    x1, y1 = (218 - crop_size) // 2, (178 - crop_size) // 2
    x2, y2 = x1 + crop_size, y1 + crop_size

    def gen():
        while True:
            cur_batch = choice(batch_names)
            images = []
            labels = []
            for name in cur_batch:
                image_path = os.path.join(celeba_dir, name)
                image = cv2.imread(image_path)
                image = image[x1:x2, y1:y2, :]
                image = cv2.resize(image, resize)
                images.append(image)
                labels.append([1])
            images = np.array(images)
            images = images / 127.5 - 1
            yield tf.convert_to_tensor(images, dtype='float'), tf.convert_to_tensor(labels)

    return gen()


# if __name__ == '__main__':
#     a = celeba_gen(32)
#     b = next(a)
#     c = (b + 1) * 127.5
#     for idx, cc in enumerate(c):
#         cv2.imwrite(r"C:\Users\jzp0306\Desktop\{0}.jpg".format(idx), cc)
