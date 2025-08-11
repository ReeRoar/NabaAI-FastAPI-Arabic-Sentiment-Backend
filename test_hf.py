import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment")

print(f"Using MODEL_ID: {MODEL_ID}")

headers = {"Authorization": f"Bearer {HF_TOKEN}"}
payload = {"inputs": "هذا منتج رائع"}

response = requests.post(f"https://api-inference.huggingface.co/models/{MODEL_ID}", headers=headers, json=payload)

print(f"Status: {response.status_code}")
print(response.json())