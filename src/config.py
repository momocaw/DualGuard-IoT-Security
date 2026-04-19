# config.py
from openai import OpenAI

# 你的 API KEY
API_KEY = "YOUR_API_KEY_HERE"

# 阿里云百炼平台的 OpenAI 兼容地址
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MODEL = "qwen3.6-plus"

def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)