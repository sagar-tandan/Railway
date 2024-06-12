from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import uvicorn
import numpy as np
from pydantic import BaseModel
from typing import List


class TextData(BaseModel):
    texts: List[str]

app = FastAPI(title="Sentiment Analysis API")


model = load_model("LSTM78percentFinal.h5")

# Load your tokenizer
tokenizer = joblib.load("tokenizerLSTM.pkl")


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/docs", status_code=308)

def get_sentiment_label(class_index):
    labels = ['Joy', 'Sadness','Inquiry', 'Neutral','Disappointment']
    return labels[class_index]

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.post("/sentiment-analysis/")
async def sentiment_analysis(data: TextData):
    results = []
    for text in data.texts:
        tokenized_text = tokenizer.texts_to_sequences([text])
        max_sequence_length = 250
        padded = pad_sequences(tokenized_text, maxlen=max_sequence_length)
        pred = model.predict(padded)
        prediction = pred.tolist()[0]
        final_prediction = prediction.index(max(prediction))
        sentiment_label = get_sentiment_label(final_prediction)
        results.append({"text": text,"predictions":prediction, "sentiment": sentiment_label})
        # results = sorted(results, key=lambda x: max(x['predictions']), reverse=True)

    return results




