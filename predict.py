from onnx_testing.predict_transformers import predict
from PIL import Image

image = Image.open("5649022222_e760db7592_z.jpg")
image = predict(image)
