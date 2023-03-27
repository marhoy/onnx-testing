from pydantic import BaseSettings


class Settings(BaseSettings):

    MODEL_CHECKPOINT = 'hustvl/yolos-tiny'
    ONNX_DIR = "hustvl_yolos_tiny_onnx"
    
