from transformers import AutoImageProcessor, AutoConfig
from PIL import Image
import torch
import numpy as np
import cv2

from onnx_testing import config
from functools import cache

@cache
def get_feature_extractor():
    return AutoImageProcessor.from_pretrained(config.MODEL_CHECKPOINT)

@cache
def get_id2label():
    return AutoConfig.from_pretrained(config.MODEL_CHECKPOINT).id2label

def parse_outputs(outputs, threshold: float, target_sizes):
    feature_extractor = get_feature_extractor()
    return feature_extractor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

def annotate_image(img_arr, results, id2label):
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(i) for i in box.tolist()]

        # Add bounding box
        arr = cv2.rectangle(img_arr, box[:2], box[2:], color=(0,0,0), thickness=2)

        # Add text
        x, y = box[:2]
        text = f"{id2label[label.item()]}: {round(score.item(), 3)}"
        arr = cv2.putText(
            arr,
            text,
            (x, y - 5),
            fontFace = cv2.FONT_HERSHEY_DUPLEX,
            fontScale = .6,
            color = (255, 255, 255),
            thickness=2
        )

    return img_arr
