from os.path import join
import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    return preprocess_input(img_array)


if __name__ == "__main__":
    hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/hot_dog'

    hot_dog_paths = [join(hot_dog_image_dir, filename) for filename in
                     ['1000288.jpg',
                      '127117.jpg']]

    not_hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/not_hot_dog'
    not_hot_dog_paths = [join(not_hot_dog_image_dir, filename) for filename in
                         ['823536.jpg',
                          '99890.jpg']]

    img_paths = hot_dog_paths + not_hot_dog_paths
