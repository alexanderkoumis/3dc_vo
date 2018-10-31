import cv2
import numpy as np


class ImageLoader(object):

    def __init__(self, max_images=3000):
        self.cache = {}
        self.max_images = max_images

    def load_image(self, image_path):
        if image_path not in self.cache:
            if len(self.cache) == self.max_images:
                self.delete_random_image()
            image = cv2.imread(image_path)
            self.cache[image_path] = image
        return self.cache[image_path]

    def delete_random_image(self):
        del_idx = np.random.randint(self.max_images)
        del_key = list(self.cache.keys())[del_idx]
        del self.cache[del_key]
