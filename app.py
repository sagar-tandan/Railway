from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import uvicorn

app = FastAPI(title="Sentiment Analysis API")
# model = load_model("LSTM78percentFinal.h5")

# Load your tokenizer
# tokenizer = joblib.load("tokenizerLSTM.pkl")
pickle_in = open("tokenizerLSTM.pkl","rb")
classifier=pickle.load(pickle_in)





@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/docs", status_code=308)

@app.get("/sentiment-analysis/{text}")
def sentiment_analysis(text: str):
    # blob = TextBlob(text)
    # polarity = blob.sentiment.polarity
    # subjectivity = blob.sentiment.subjectivity

    # if polarity > 0:
    #     sentiment = "positive"
    # elif polarity < 0:
    #     sentiment = "negative"
    # else:
    #     sentiment = "neutral"
        
    # tokenized_text = tokenizer.texts_to_sequences([text])
    # max_sequence_length = 250
    # tokenized_text = pad_sequences(tokenized_text, maxlen=max_sequence_length, padding='post')
    # prediction = model.predict(tokenized_text)
    prediction = "Tokenizer Loaded"





    return {
        "text": text,
        # "sentiment": sentiment,
        # "polarity": polarity,
        # "subjectivity": subjectivity,
        "prediction" : prediction,
    }
