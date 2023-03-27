#%%
from transformers import pipeline
from PIL import Image
import requests

pipe = pipeline("object-detection", "hustvl/yolos-tiny")

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

print(pipe(image))
# %%
