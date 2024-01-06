import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import platform
import time
import pathlib
import os
#importing required libraries
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import jenkspy
import json
import streamlit as st
import numpy as np
import wave
import struct
import numpy as np
import ast
import base64
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

#S R1 R2 G2 G3 M1 M2 P D1 D2 N2 N3 
#0 1 2 3 4 5 6 7 8 9 10 11

#S_4 R1_4 R2_4 G2_4 G3_4 M1_4 M2_4 P_4 D1_4 D2_4 N2_4 N3_4
#0 14 2 16 4 6 18 8 20 10 22 12  

########################### UTILITY FUNCTION ############################
note_to_number={
    'S_4': 0,
    'S_5': 1,
    'R2_4': 2,
    'R2_5': 3,
    'G3_4': 4,
    'G3_5': 5,
    'M1_4': 6,
    'M1_5': 7,
    'P_4': 8,
    'P_5': 9,
    'D2_4': 10,
    'D2_5': 11,
    'N3_4': 12,
    'N3_5': 13,
    'R1_4': 14,
    'R1_5': 15,
    'G2_4': 16,
    'G2_5': 17,
    'M2_4': 18,
    'M2_5': 19,
    'D1_4': 20,
    'D1_5': 21,
    'N2_4': 22,
    'N2_5': 23
}

def filter_notes(notes_array, note_to_number_dict):
    filtered_notes = [note for note in notes_array if note in note_to_number_dict]
    return filtered_notes


def numbers_to_notes_reverse(number_array, note_to_number):
    reversed_notes = []

    # Reverse the array of numbers
    reversed_numbers = reversed(number_array)

    # Convert each number back to its corresponding note
    for number in reversed_numbers:
        for note, value in note_to_number.items():
            if value == number:
                reversed_notes.append(note)
                break

    # Print the reversed notes
    st.write("Reversed Notes:", reversed_notes)
    return reversed_notes





# def numbers_to_notes(number_array):
#     notes_array = [key for key, value in note_to_number.items() if value in number_array]
#     return notes_array
def numbers_to_notes(numbers_array):
    number_to_note = {v: k for k, v in note_to_number.items()}
    notes_array = [number_to_note[num] for num in numbers_array]
    return notes_array
def notes_to_numbers(notes_array):
    number_array = [note_to_number[note] for note in notes_array]
    return number_array



def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href
############################## Initialize ##################################


# Some Useful Variables
window_size = 2205    # Size of window to be used for detecting silence
beta = 1   # Silence detection parameter
max_notes = 10000    # Maximum number of notes in file, for efficiency
sampling_freq = 44100	# Sampling frequency of audio signal
threshold = 1600


#C C# D D# E F F# G G# A A# B
#Sa Ri1 Ri2/Ga1 Ri3/Ga2 Ga3 Ma1 Ma2 Pa Dha1 Dha2/Ni1 Dha3/Ni2 Ni3


# Array for musical notes and their notations

# C C# D D# E F F# G G# A A# B
# Sa Ri1 Ri2/Ga1 Ri3/Ga2 Ga3 Ma1 Ma2 Pa Dha1 Dha2/Ni1 Dha3/Ni2 Ni3

notes = [
    'S_0', 'R1_0', 'R2_0', 'G2_0', 'G3_0', 'M1_0', 'M2_0', 'P_0', 'D1_0', 'D2_0', 'N2_0', 'N3_0',
    'S_1', 'R1_1', 'R2_1', 'G2_1', 'G3_1', 'M1_1', 'M2_1', 'P_1', 'D1_1', 'D2_1', 'N2_1', 'N3_1',
    'S_2', 'R1_2', 'R2_2', 'G2_2', 'G3_2', 'M1_2', 'M2_2', 'P_2', 'D1_2', 'D2_2', 'N2_2', 'N3_2',
    'S_3', 'R1_3', 'R2_3', 'G2_3', 'G3_3', 'M1_3', 'M2_3', 'P_3', 'D1_3', 'D2_3', 'N2_3', 'N3_3',
    'S_4', 'R1_4', 'R2_4', 'G2_4', 'G3_4', 'M1_4', 'M2_4', 'P_4', 'D1_4', 'D2_4', 'N2_4', 'N3_4',
    'S_5', 'R1_5', 'R2_5', 'G2_5', 'G3_5', 'M1_5', 'M2_5', 'P_5', 'D1_5', 'D2_5', 'N2_5', 'N3_5',
    'S_6', 'R1_6', 'R2_6', 'G2_6', 'G3_6', 'M1_6', 'M2_6', 'P_6', 'D1_6', 'D2_6', 'N2_6', 'N3_6',
    'S_7', 'R1_7', 'R2_7', 'G2_7', 'G3_7', 'M1_7', 'M2_7', 'P_7', 'D1_7', 'D2_7', 'N2_7', 'N3_7',
    'S_8', 'R1_8', 'R2_8', 'G2_8', 'G3_8', 'M1_8', 'M2_8', 'P_8', 'D1_8', 'D2_8', 'N2_8', 'N3_8'
]


# Array for corresponding frequencies in Hertz
array = [
    16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87,
    32.70, 34.65, 36.71, 38.89, 41.20, 43.65, 46.25, 49.00, 51.91, 55.00, 58.27, 61.74,
    65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47,
    130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94,
    261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88,
    523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77,
    1046.50, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760.00, 1864.66, 1975.53,
    2093.00, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520.00, 3729.31, 3951.07,
    4186.01,4434.92,4698.63,4978.03,5274.04,5587.65,5919.91,6271.93,6644.88,7040.00,7458.62,7902.13
      
]
Identified_Notes = []


#title of page
st.title("Raagdhvani 🎵")
st.header("Classify your song")                                                                                                                                                  

#disabling warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

#hiding menu
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)


from io import StringIO

uploaded_file = st.file_uploader("Choose a file to classify (wav/mp3)")

if uploaded_file is not None:
    ############################## Read Audio File #############################
    st.write('\n\nReading Audio File...')

    sound_file = wave.open(uploaded_file, 'r')
    file_length = sound_file.getnframes()
    print(sound_file.getparams())
    print(file_length)
    sound = np.zeros(file_length)
    mean_square = []
    sound_square = np.zeros(file_length)
    for i in range(file_length):
        data = sound_file.readframes(1)
        data = struct.unpack("hh", data)
        sound[i] = int(data[0])

    #print(sound)
    sound = np.divide(sound, float(2**15))	# Normalize data in range -1 to 1
    #st.write("after normlaizing\n")
    #st.write(sound)
    ######################### DETECTING SCILENCE ##################################

    sound_square = np.square(sound)
    frequency = []
    dft = []
    i = 0
    j = 0
    k = 0    
    # traversing sound_square array with a fixed window_size
    while(i<=len(sound_square)-window_size):
        s = 0.0
        j = 0
        while(j<=window_size):
            s = s + sound_square[i+j]
            j = j + 1	
    # detecting the silence waves
        if s < threshold:
            if(i-k>window_size*4):
                dft = np.array(dft) # applying fourier transform function
                dft = np.fft.fft(sound[k:i])
                dft=np.argsort(dft)

                if(dft[0]>dft[-1] and dft[1]>dft[-1]):
                    i_max = dft[-1]
                elif(dft[1]>dft[0] and dft[-1]>dft[0]):
                    i_max = dft[0]
                else :	
                    i_max = dft[1]
    # claculating frequency				
                frequency.append((i_max*sampling_freq)/(i-k))
                dft = []
                k = i+1
        i = i + window_size

     

    for i in frequency :
        print(i)
        idx = (np.abs(array-i)).argmin()
        Identified_Notes.append(notes[idx])
    #st.write(Identified_Notes)


    st.write("IDENTIFIED NOTES\n")
    # Convert the output array to a comma-separated string
    output_string = ', '.join(Identified_Notes)
    st.write(output_string)

    filtered_identified_notes=filter_notes(Identified_Notes,note_to_number)
    #st.write(filtered_identified_notes)


    st.write("FILTERED IDENTIFIED NOTES WITHIN OCTAVES 4 AND 5\n")
    # Convert the output array to a comma-separated string
    filtered_output_string = ', '.join(filtered_identified_notes)
    st.write(filtered_output_string)

    
    
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
    # 'S_4': 0,
    # 'S_5': 1,
    # 'R2_4': 2,
    # 'R2_5': 3,
    # 'G3_4': 4,
    # 'G3_5': 5,
    # 'M1_4': 6,
    # 'M1_5': 7,
    # 'P_4': 8,
    # 'P_5': 9,
    # 'D2_4': 10,
    # 'D2_5': 11,
    # 'N3_4': 12,
    # 'N3_5': 13,
    # 'R1_4': 14,
    # 'R1_5': 15,
    # 'G2_4': 16,
    # 'G2_5': 17,
    # 'M2_4': 18,
    # 'M2_5': 19,
    # 'D1_4': 20,
    # 'D1_5': 21,
    # 'N2_4': 22,
    # 'N2_5': 23
    shankarabharanam_notes = [2,3,4,5,6,7,10,11,12,13]
    bhavapriya_notes = [14,15,16,17,18,19,20,21,22,23]


    from tensorflow.keras.models import Model


    from tensorflow.keras.models import load_model

    # Load the saved model
    model = load_model('notereductionlstmmodel.keras')
    #st.write('Converted Notes to numbers\n')
    converted_notes=notes_to_numbers(filtered_identified_notes)
    #st.write(converted_notes)
    new_text_sequence=converted_notes
    padded_new_text_sequence = pad_sequences([new_text_sequence], maxlen=max_text_length, padding='post')
    predicted_summary = model.predict(padded_new_text_sequence)

    # Convert the predicted summary back to integer values
    predicted_summary = [np.argmax(seq, axis=1) for seq in predicted_summary]
    predicted_summary_final = predicted_summary[0]

    # Initialize variables to keep track of the likelihood for each raga
    likelihood_shankarabharanam = 0
    likelihood_bhavapriya = 0

    # Iterate over each number in the predicted summary
    for note in predicted_summary_final:
        # Check if the note is in the Shankarabharanam or Bhavapriya lists
        if note in shankarabharanam_notes:
            likelihood_shankarabharanam += 1
        if note in bhavapriya_notes:
            likelihood_bhavapriya += 1

    # Calculate the likelihood scores as a ratio of the total notes
    total_notes = len(predicted_summary_final)
    likelihood_shankarabharanam /= total_notes
    likelihood_bhavapriya /= total_notes
    print(likelihood_shankarabharanam)
    print(likelihood_bhavapriya)

    # Determine which raga has a higher likelihood
    predicted_raga = "Shankarabharanam" if likelihood_shankarabharanam > likelihood_bhavapriya else "Bhavapriya"
    st.write("Predicted Raga: "+predicted_raga)
    st.write("Predicted REDUCED NOTES: ")
    #st.write(predicted_summary_final)
    


    from tensorflow.keras.models import load_model

    # Load the saved model
    model = load_model('notegenerationlstmmodel.keras')
    # For example:
    new_sequence = np.array([predicted_summary_final], dtype=np.float32)
  
    st.write(new_sequence)
    new_sequence = new_sequence.reshape((new_sequence.shape[0], 1, new_sequence.shape[1]))

    prediction = model.predict(new_sequence)

    # Round the predictions to the nearest integer
    rounded_prediction = np.round(prediction).astype(int)



    st.write("GENERATION OUTPUT")
    st.write("Predicted Succeeding Sequence:\n")
    st.write(rounded_prediction)


    output_numbers=input_numbers=np.array(rounded_prediction).flatten()

    final_composition=numbers_to_notes(output_numbers)
    st.write(final_composition)



    import numpy as np
    from pydub import AudioSegment
    from datetime import datetime

    def generate_wave(note, duration, amplitude=0.5, sample_rate=44100):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        wave = amplitude * np.sin(2 * np.pi * note * t)
        # Normalize the wave to fit within the range [-1, 1]
        wave = np.int16(wave * 32767)
        return wave

    def create_wav_file(notes, durations, output_file, amplitude=0.8, sample_rate=44100, sample_width=2):
        combined = AudioSegment.silent(duration=0)

        for note, duration in zip(notes, durations):
            wave = generate_wave(note, duration, amplitude, sample_rate)
            audio_segment = AudioSegment(
                wave.tobytes(),
                frame_rate=sample_rate,
                sample_width=sample_width,
                channels=1
            )
            combined += audio_segment

        combined.export(output_file, format="wav")

    def get_frequency(note):
        note_mapping = {
        'S_4': 261.63,
        'R1_4': 277.18,
        'R2_4': 293.66,
        'G2_4': 311.13,
        'G3_4': 329.63,
        'M1_4': 349.23,
        'M2_4': 369.99,
        'P_4': 392.00,
        'D1_4': 415.30,
        'D2_4': 440.00,
        'N2_4': 466.16,
        'N3_4': 493.88,
        'S_5': 523.25,
        'R1_5': 554.37,
        'R2_5': 587.33,
        'G2_5': 622.25,
        'G3_5': 659.25,
        'M1_5': 698.46,
        'M2_5': 739.99,
        'P_5': 783.99,
        'D1_5': 830.61,
        'D2_5': 880.00,
        'N2_5': 932.33,
        'N3_5': 987.77
    }


        return note_mapping[note]

    def generate_notes_array(result_string):
        notes = []
        durations = []
        for note in result_string:
                frequency = get_frequency(note)
                notes.append(frequency)
                durations.append(0.5)  # Assuming a default duration of 0.5 seconds for each note
        print(notes)
        print(durations)
        return notes, durations


    from datetime import datetime

    # Your existing code for generating notes and creating the WAV file
    result_string = final_composition
    notes, durations = generate_notes_array(final_composition)

    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H%M%S")
    output_file = "genskb"+dt_string+".wav"
    create_wav_file(notes, durations, output_file, amplitude=0.8)

    st.audio(output_file, format="audio/wav")
    from datetime import datetime

    
    
    st.markdown(get_binary_file_downloader_html("genskb"+dt_string+".wav", 'WAV'), unsafe_allow_html = True)






























    