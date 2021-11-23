import torch

from config import config
from models.experimental import attempt_load
from service.convert_image import ConvertImage
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device
import base64
import cv2


def transform_data(labels, boxes, image):
    # if labels.count() == 0:
    #     return {'message': 'please take the foot image'}
    box = boxes[0]
    transformed_data = {'label': labels[0], 'width': {
        'start': box[0][0],
        'end': box[1][0]
    }, 'height': {
        'start': box[0][1],
        'end': box[1][1]
    }, 'image': image}
    return transformed_data


def get_prediction_box(pred, img0, img, names):
    predict_labels = []
    predict_boxes = []
    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = names[int(cls)]
                predict_labels.append(label)
                predict_boxes.append(plot_one_box(xyxy))
    return predict_labels, predict_boxes


class DetectImage:
    def __init__(self, image):
        self.image = image

    def detect(self):
        # Initialize
        set_logging()
        device = select_device('')
        model = attempt_load(config.WEIGHTS, map_location=device)  # load FP32 model
        image_size = check_img_size(config.IMG_SIZE, s=model.stride.max())  # check img_size
        img, img0 = ConvertImage(self.image, image_size).base64_to_image()
        names = model.module.names if hasattr(model, 'module') else model.names
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=config.AUGMENT)[0]
        pred = non_max_suppression(pred, config.CONF, config.IOU_THRES, classes=config.CLASSES,
                                   agnostic=config.ASNOSTIC_NMS)

        labels, predict_boxes = get_prediction_box(pred, img0, img, names)
        image = self.crop_image(img0, predict_boxes)
        return transform_data(labels, predict_boxes, image)

    def crop_image(self, image, predict_boxes):
        box = predict_boxes[0]
        print(box)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop_image = rgb_image[box[0][1]:box[1][1],box[0][0]:box[1][0]]
        _, im_arr = cv2.imencode('.jpg', crop_image)
        im_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(im_bytes).decode("utf-8")

        return im_b64

