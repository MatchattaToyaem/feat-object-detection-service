import base64
import io
import cv2
import numpy as np
from PIL.Image import Image

from utils.datasets import letterbox


class ConvertImage:
    def __init__(self, image, image_size):
        self.image_data = image
        self.image_size = image_size

    def base64_to_image(self):
        im_bytes = base64.b64decode(self.image_data.split(',')[1])
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_UNCHANGED)
        img0 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = letterbox(img0, new_shape=self.image_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img, img0



