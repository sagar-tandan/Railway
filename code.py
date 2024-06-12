# All the codes were written in tested in kaggle and it it just the copy of that code
# if you want to run the code just paste the while code in kaggle notebook 

import pandas as pd
import numpy as np
# text preprocessing
from nltk.tokenize import word_tokenize
import re


import urllib.request
import zipfile
import os

# plots and metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# preparing input to our model
#from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# keras layers
from keras.models import Sequential
from keras.layers import Embedding,Bidirectional, GRU,Conv1D, SpatialDropout1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout, GlobalMaxPooling1D,GlobalAveragePooling1D
from keras.layers import concatenate
from keras.optimizers import Adam

from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping



from sklearn.model_selection import train_test_split


# Number of labels: joy, anger, fear, sadness, neutral
num_classes = 5

# Number of dimensions for word embedding
embed_num_dims = 300

# Max input length (max number of words)
max_seq_len = 250

class_names = ['Joy', 'Sadness','Inquiry', 'Neutral','Disappointment']

import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('/kaggle/input/datasetfinal/ytYogiNeutral.csv')

print(df.Emotion.value_counts())
df = df[df['Emotion'] != 'Humor']
df = df[df['Emotion'] != 'Surprise']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Comment'], df['Emotion'], test_size=0.2, random_state=42)

print(df.Emotion.value_counts())
df.head(6)

def clean_text(data):

    # remove hashtags and @usernames
    data = re.sub(r'@[A-Za-zA-Z0-9]+', '', data)  # removing @mentions
    data = re.sub(r'@[A-Za-z]+', '', data)        # removing @mentions
    data = re.sub(r'@[-)]+', '', data)            # removing @mentions

    # tekenization using nltk
    data = word_tokenize(data)

    return data


nltk.download('punkt')

texts = [' '.join(clean_text(str(text))) for text in df.Comment]
texts_train = [' '.join(clean_text(str(text))) for text in X_train]
texts_test = [' '.join(clean_text(str(text))) for text in X_test]

#Tokenize

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequence_train = tokenizer.texts_to_sequences(texts_train)
sequence_test = tokenizer.texts_to_sequences(texts_test)

index_of_words = tokenizer.word_index

# vocab size is number of unique words + reserved 0 index for padding
vocab_size = len(index_of_words) + 1

print('Number of unique words: {}'.format(len(index_of_words)))

#Padding

X_train_pad = pad_sequences(sequence_train, maxlen = max_seq_len )
X_test_pad = pad_sequences(sequence_test, maxlen = max_seq_len )

X_train_pad

encoding = {
    'Joy': 0,
    'Sadness': 1,
    'Inquiry': 2,
    'Neutral': 3,
    'Disappointment': 4

}

# Integer labels
#y_train = [encoding[x] for x in data_train.Emotion]
#y_test = [encoding[x] for x in data_test.Emotion]



# Convert emotion labels to integer labels using the encoding dictionary
y_train = [encoding[x] for x in y_train]
y_test = [encoding[x] for x in y_test]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix

fname = 'embeddings/wiki-news-300d-1M.vec'

if not os.path.isfile(fname):
    print('Downloading word vectors...')
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
                              'wiki-news-300d-1M.vec.zip')
    print('Unzipping...')
    with zipfile.ZipFile('wiki-news-300d-1M.vec.zip', 'r') as zip_ref:
        zip_ref.extractall('embeddings')
    print('done.')

    os.remove('wiki-news-300d-1M.vec.zip')
    
    embedd_matrix = create_embedding_matrix(fname, index_of_words, embed_num_dims)
embedd_matrix.shape

# Inspect unseen words
new_words = 0

for word in index_of_words:
    entry = embedd_matrix[index_of_words[word]]
    if all(v == 0 for v in entry):
        new_words = new_words + 1


# Define your model
model = Sequential()
model.add(Embedding(vocab_size, 300, input_length = 250, weights=[embedd_matrix],trainable= False))  # Adjust input length here
model.add(Dropout(0.4))
model.add((LSTM(64,dropout = 0.4,return_sequences= True)))
model.add(Dropout(0.4))
model.add((LSTM(40,dropout = 0.4)))
model.add(Dropout(0.4))
model.add(Dense(50, activation='softmax'))
model.add(Dropout(0.4))
model.add(Dense(5, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.summary()


history = model.fit(X_train_pad, y_train,batch_size = 32,epochs=100, validation_data=(X_test_pad, y_test),callbacks=[early_stopping])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# Assuming 'history' is the object returned by model.fit()
train_losses = history.history['loss']  # Training loss values for each epoch
val_losses = history.history['val_loss']  # Validation loss values for each epoch

# Calculate errors (difference between predicted and actual loss)
train_errors = np.array(train_losses) - np.array(val_losses)

# Calculate variance of the errors
variance_error = np.var(train_errors, ddof=1)  # ddof=1 for unbiased estimation

print("Variance of the errors:", variance_error)



from tensorflow.keras.models import Model

# Create a new model for predictions
prediction_model = Model(inputs=model.input, outputs=model.output)

# Make predictions
predictions = prediction_model.predict(X_test_pad)


# Convert predictions to class labels
predicted_labels = [np.argmax(pred) for pred in predictions


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix



# Convert one-hot encoded true labels to class labels
true_labels = [np.argmax(label) for label in y_test]

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)

# Calculate F1 score
f1 = f1_score(true_labels, predicted_labels, average='weighted')
print("F1 Score:", f1)


from sklearn.metrics import classification_report
print(classification_report(true_labels, predicted_labels))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
 

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    
    # Set size
    fig.set_size_inches(12.5, 7.5)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.grid(False)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
    
    # Plot normalized confusion matrix
plot_confusion_matrix(true_labels, predicted_labels, classes=class_names, normalize=True, title='Normalized confusion matrix')
plt.show()

model.save('LSTM78percentFinal.h5')
import pickle
pickle.dump(tokenizer,open("tokenizerLSTM.pkl","wb"))

from keras.models import load_model
model=load_model("/kaggle/input/modelprediction/YTSentimentModel83Acc.h5")
import pickle
label_encoder= pickle.load(open("/kaggle/input/modelprediction/tokenizer.pkl",'rb'))

import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('/kaggle/working/LSTM78percentFinal.h5')

# Load the tokenizer
with open('/kaggle/working/tokenizerLSTM.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
    
    
    
    import time

message = ['i need ciggerates .']
seq = tokenizer.texts_to_sequences(message)
padded = pad_sequences(seq, maxlen=max_seq_len)

start_time = time.time()

pred = model.predict(padded)

print('Message: ' + str(message))
print('predicted: {} ({:.2f} seconds)'.format(class_names[np.argmax(pred)], (time.time() - start_time)))


    