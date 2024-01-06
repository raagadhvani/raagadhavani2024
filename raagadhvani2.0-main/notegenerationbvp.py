# import numpy as np
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from ast import literal_eval

# # Load data from CSV file
# df = pd.read_csv('FinalMergedNumberBVP.csv')  # Replace 'your_dataset.csv' with the actual filename

# # Convert string representations of lists to actual lists
# df['Input Sequence'] = df['Input Sequence'].apply(literal_eval)
# df['Succeeding Sequence'] = df['Succeeding Sequence'].apply(literal_eval)

# # Convert sequences to numpy arrays
# input_sequences = np.array(df['Input Sequence'].tolist(), dtype=np.float32)
# succeeding_sequences = np.array(df['Succeeding Sequence'].tolist(), dtype=np.float32)

# # Reshape the input sequences to match LSTM input shape (samples, time steps, features)
# input_sequences = input_sequences.reshape((input_sequences.shape[0], 1, input_sequences.shape[1]))

# # Build the LSTM model
# model = Sequential()
# model.add(LSTM(50, activation='relu', input_shape=(1, 8)))
# model.add(Dense(8))  # Output layer with 8 units since your sequences are of length 8

# # Compile the model
# model.compile(optimizer='adam', loss='mse')  # You can adjust the optimizer and loss based on your problem

# # Train the model
# model.fit(input_sequences, succeeding_sequences, epochs=1000, verbose=2)


# #Save Model
# model.save("notegenerationlstmmodelbvp.keras")

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from ast import literal_eval

# Load data from CSV file
df = pd.read_csv('FinalMergedNumberBVP.csv')  # Replace 'your_dataset.csv' with the actual filename

# Convert string representations of lists to actual lists
df['Input Sequence'] = df['Input Sequence'].apply(literal_eval)
df['Succeeding Sequence'] = df['Succeeding Sequence'].apply(literal_eval)

# Convert sequences to numpy arrays
input_sequences = np.array(df['Input Sequence'].tolist(), dtype=np.float32)
succeeding_sequences = np.array(df['Succeeding Sequence'].tolist(), dtype=np.float32)

# Reshape the input sequences to match LSTM input shape (samples, time steps, features)
input_sequences = input_sequences.reshape((input_sequences.shape[0], 1, input_sequences.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 8)))
model.add(Dense(8, activation='relu'))  # Apply ReLU activation to the Dense layer
model.add(Dense(8))  # Output layer with 8 units since your sequences are of length 8

# Compile the model
model.compile(optimizer='adam', loss='mse')  # You can adjust the optimizer and loss based on your problem

# Train the model
model.fit(input_sequences, succeeding_sequences, epochs=1000, verbose=2)

# Save Model
model.save("notegenerationlstmmodelbvp.keras")
