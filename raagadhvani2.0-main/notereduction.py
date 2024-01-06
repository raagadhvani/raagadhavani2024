import numpy as np
import ast
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

df=pd.read_csv('notesNumbersUPD.csv')
# Sample data (replace with your actual dataset)
text_sequences =  [ast.literal_eval(data_str) for data_str in df['numerical_notes_text']]

summary_sequences = [ast.literal_eval(data_str) for data_str in df['numerical_notes_summary']]


# Preprocess the data
max_text_length = max(len(seq) for seq in text_sequences)
max_summary_length = max(len(seq) for seq in summary_sequences)

# Padding the text sequences and one-hot encoding the summary sequences
padded_text_sequences = pad_sequences(text_sequences, maxlen=max_text_length, padding='post')
padded_summary_sequences = pad_sequences(summary_sequences, maxlen=max_summary_length, padding='post')
#df=pd.read_csv('notesUPD.csv')

# Assuming you have a DataFrame with a column 'expected_raga' that contains the expected ragas
expected_ragas = df['raga'].tolist()

input_sequences = df['numerical_notes_text']

# Initialize lists to store the predicted ragas and likelihood scores
predicted_ragas = []
likelihood_shankarabharanam_scores = []
likelihood_bhavapriya_scores = []

# Define the lists of notes for Shankarabharanam and Bhavapriya
shankarabharanam_notes = [2, 4, 5, 9, 11]
bhavapriya_notes = [1, 3, 6, 8, 10]


# Define input layer
input_layer = Input(shape=(max_text_length,))
embedding_layer = Embedding(input_dim=max_text_length, output_dim=128)(input_layer)

# Encoder LSTM
encoder_lstm = LSTM(64)(embedding_layer)

# Create a RepeatVector layer to match the sequence length of the summary
repeat_vector = tf.keras.layers.RepeatVector(max_summary_length)(encoder_lstm)

# Decoder LSTM
decoder_lstm = LSTM(128, return_sequences=True)(repeat_vector)
vocab_size = 23
#len(set([word for sequence in summary_sequences for word in sequence]))
output_layer = Dense(vocab_size, activation='softmax')(decoder_lstm)

# # Output layer
# output_layer = Dense(max_text_length, activation='softmax')(decoder_lstm)

# Create the model using the functional API
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train the model
history=model.fit(padded_text_sequences, padded_summary_sequences, epochs=1000)

# # Save the trained model to a file
model.save("notereductionlstmmodel.keras")