from onnx_testing import config
from onnx_testing.predict_common import get_feature_extractor, parse_outputs, annotate_image

import torch
from transformers import AutoModelForObjectDetection
from transformers.models.yolos.modeling_yolos import YolosObjectDetectionOutput

def predict(image, threshold: float = 0.9):
    feature_extractor = get_feature_extractor()
    inputs = feature_extractor(images=image, return_tensors="pt")

    model = AutoModelForObjectDetection.from_pretrained(config.MODEL_CHECKPOINT)
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = parse_outputs(outputs, threshold=threshold, target_sizes=target_sizes)

    return annotate_image(image, results)
