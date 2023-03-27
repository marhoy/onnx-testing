
from onnx_testing import config
from onnx_testing.predict_common import annotate_image
from onnxruntime import InferenceSession
from transformers import YolosImageProcessor, AutoConfig
import torch
from transformers.models.yolos.modeling_yolos import YolosObjectDetectionOutput

feature_extractor = YolosImageProcessor.from_pretrained(config.ONNX_DIR)
session = InferenceSession(config.ONNX_DIR + "/model.onnx")
id2label = AutoConfig.from_pretrained(config.ONNX_DIR).id2label

def predict(img_arr, threshold: float = 0.9):
    inputs = feature_extractor(images=img_arr, return_tensors="np")
    output_list = session.run([], input_feed=dict(inputs))

    outputs = YolosObjectDetectionOutput(
            logits=torch.FloatTensor(output_list[0]),
            pred_boxes=torch.FloatTensor(output_list[1])
    )

    target_sizes = torch.tensor([img_arr.shape[:2]])
    results = feature_extractor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    return annotate_image(img_arr, results, id2label)
