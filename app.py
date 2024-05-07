from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import uvicorn
import numpy as np
from pydantic import BaseModel

class TextData(BaseModel):
    texts: list[str]

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
    return results


# @app.get("/sentiment-analysis/{text}")
# def sentiment_analysis(text: str):
#     # blob = TextBlob(text)
#     # polarity = blob.sentiment.polarity
#     # subjectivity = blob.sentiment.subjectivity

#     # if polarity > 0:
#     #     sentiment = "positive"
#     # elif polarity < 0:
#     #     sentiment = "negative"
#     # else:
#     #     sentiment = "neutral"
        
#     tokenized_text = tokenizer.texts_to_sequences([text])
#     max_sequence_length = 250

#     padded = pad_sequences(tokenized_text, maxlen=max_sequence_length)
#     pred = model.predict(padded)
#     prediction = pred.tolist()[0]
#     finalPrediction = prediction.index(max(prediction))
#     sentiment_label = get_sentiment_label(finalPrediction)


#     prediction = model.summary()





#     return {
#         "text": text,
#         "prediction" : pred.tolist()[0],
#         "index" : finalPrediction,
#         "Emotion" : sentiment_label,
#     }
