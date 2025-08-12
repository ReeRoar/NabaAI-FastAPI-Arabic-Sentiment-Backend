from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
#frontend URL http://localhost:57185
app = FastAPI()

# Load model and tokenizer once at startup
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "API is running locally"}

@app.post("/predict")
def predict_sentiment(input: TextInput):
    try:
        inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits).item()

        #  labels are dynamic according to the model not hardcoded :))
        id2label = model.config.id2label
        predicted_label = id2label[predicted_class_id]

        return {"input": input.text, "sentiment": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
#frontend URL to avoid XMLHttpRequest failures
origins=["*"]
app.add_middleware (
    CORSMiddleware,
    allow_origins= ["*"], #allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)