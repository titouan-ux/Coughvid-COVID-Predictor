# This script includes adapted code from:
# https://github.com/skanderhamdi/attention_cnn_lstm_covid_mel_spectrogram
# Original license and authorship belong to the original contributors

# Import libraries =============================================================
import pandas as pd
import os
import librosa
import librosa.display
import cv2
import numpy as np
import soundfile as sf


# Pitch shifting ===============================================================
def pitchShift(metaDataPath, audioDataPath, augmentedSignals):

    metaData = pd.read_csv(metaDataPath)

    counter = 0
    for index, row in metaData.iterrows():
        fname = row["non_silent_name"]
        if (index + 1) % 100 == 0:
            print(fname, " ", str(index + 1), "/", str(metaData.shape[0]))
        signal, sr = librosa.load(audioDataPath + fname)

        # For COVID-19 samples
        if row["label"] == "COVID-19" or row["label"] == "symptomatic":

            # Save original audio
            sf.write(
                augmentedSignals + "sample{0}_{1}.wav".format(counter, 1),
                signal,
                sr,
                "PCM_24",
            )
            counter += 1
            # Pitch shift down 4 steps
            pitch_shifting = librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=-4)
            # Save pitch shifted audio
            sf.write(
                augmentedSignals + "sample{0}_{1}.wav".format(counter, 1),
                pitch_shifting,
                sr,
                "PCM_24",
            )
            counter += 1

        # For healthy samples
        else:
            # Only save original audio
            sf.write(
                augmentedSignals + "sample{0}_{1}.wav".format(counter, 0),
                signal,
                sr,
                "PCM_24",
            )
            counter += 1
