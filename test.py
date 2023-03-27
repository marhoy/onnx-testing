
#%%
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image

image = Image.open("test.jpg")

feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

# model predicts bounding boxes and corresponding COCO classes
logits = outputs.logits
bboxes = outputs.pred_boxes

#%%
from onnxruntime import InferenceSession
from transformers import YolosImageProcessor
from PIL import Image

model_dir = "onnx-transf"
processor = YolosImageProcessor.from_pretrained(model_dir)
session = InferenceSession(model_dir + "/model.onnx")

image = Image.open("5649022222_e760db7592_z.jpg")

inputs = processor(images=image, return_tensors="np")
outputs = session.run([], input_feed=dict(inputs))
# #output_names=["last_hidden_state"], input_feed=inputs["pixel_values"])
logits, bbox = outputs

#%%
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTM

tokenizer = YolosImageProcessor.from_pretrained("onnx")
model = ORTModel.from_pretrained("onnx")
inputs = tokenizer(images=image, return_tensors="pt")
outputs = model(**inputs)
