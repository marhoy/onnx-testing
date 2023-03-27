
from onnx_testing import config
from onnx_testing.predict_common import parse_outputs, annotate_image
from onnxruntime import InferenceSession
from transformers import YolosImageProcessor
import torch
from transformers.models.yolos.modeling_yolos import YolosObjectDetectionOutput

feature_extractor = YolosImageProcessor.from_pretrained(config.ONNX_DIR)
session = InferenceSession(config.ONNX_DIR + "/model.onnx")

def predict(image, threshold: float = 0.9):
    inputs = feature_extractor(images=image, return_tensors="np")
    output_list = session.run([], input_feed=dict(inputs))
    
    outputs = YolosObjectDetectionOutput(logits=torch.FloatTensor(output_list[0]), pred_boxes=torch.FloatTensor(output_list[1]))


    target_sizes = torch.tensor([image.size[::-1]])
    results = parse_outputs(outputs, threshold=threshold, target_sizes=target_sizes)

    return annotate_image(image, results)
