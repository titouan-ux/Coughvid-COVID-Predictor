# This script includes adapted code from:
# https://github.com/skanderhamdi/attention_cnn_lstm_covid_mel_spectrogram
# Original license and authorship belong to the original contributors

import pandas as pd
import numpy as np
import librosa
import os
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow_io as tfio

## Function for showing a progress bar


def progressBar(
    iterable, prefix="", suffix="", decimals=1, length=100, fill="█", printEnd="\r"
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)

    # Progress Bar Printing Function
    def printProgressBar(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(
            100 * (iteration / float(total))
        )
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + "-" * (length - filledLength)
        print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)

    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()


# Function for augmenting audio and creating melspectograms ====================
def SpectAugment(waves_path, files, param_masking, mels_path, labels_path, mean_signal_length):
  
    labels_list = []
    count = 0
    meanSignalLength = mean_signal_length
    
    for fn in progressBar(files, prefix = 'Converting:', suffix = '', length = 50):
        if fn == '.DS_Store':
            continue
        label = fn.split('.')[0].split('_')[1]
        signal , sr = librosa.load(waves_path+fn)
        s_len = len(signal)
        
        # Add zero padding to the signal if less than 156027 (~4.07 seconds)
        # Remove from begining and the end if signal length is greater than 156027 (~4.07 seconds)
        if s_len < meanSignalLength:
               pad_len = meanSignalLength - s_len
               pad_rem = pad_len % 2
               pad_len //= 2
               signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
        else:
               pad_len = s_len - meanSignalLength
               pad_len //= 2
               signal = signal[pad_len:pad_len + meanSignalLength]
        label = fn.split('.')[0].split('_')[1]
        mel_spectrogram = librosa.feature.melspectrogram(y=signal,sr=sr,n_mels=128,hop_length=512,fmax=8000,n_fft=512,center=True)
        dbscale_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max,top_db=80)
        
        plt.figure(figsize=(5.15, 1.99), dpi=100)  # Dimensions: 515 x 199
        plt.imshow(dbscale_mel_spectrogram, interpolation='nearest', origin='lower', aspect='auto')
        plt.axis('off')
        plt.savefig(mels_path + str(count) + ".png", dpi=100)
        plt.close()
        img = Image.open(mels_path + str(count) + ".png").convert("RGB")
        img.save(mels_path + str(count) + ".png")
        
        # Save image names with corresponding labels (0: no COVID, 1: COVID)
        labels_list.append({'filename': f"{count}.png", 'label': label})
        count+=1
        
        if label == '1': # if COVID
            freq_mask = tfio.audio.freq_mask(dbscale_mel_spectrogram, param=param_masking)
            time_mask = tfio.audio.time_mask(freq_mask, param=param_masking)

            plt.figure(figsize=(5.15, 1.99), dpi=100)  # 515 x 199
            plt.imshow(time_mask, interpolation='nearest', origin='lower', aspect='auto')
            plt.axis('off')
            plt.savefig(mels_path + str(count) + ".png", dpi=100)
            plt.close()
            img = Image.open(mels_path + str(count) + ".png").convert("RGB")
            img.save(mels_path + str(count) + ".png")
            
            # Save image names with corresponding labels (0: no COVID, 1: COVID)
            labels_list.append({'filename': f"{count}.png", 'label': label})
            count+=1

        freq_mask = tfio.audio.freq_mask(dbscale_mel_spectrogram, param=param_masking)
        time_mask = tfio.audio.time_mask(freq_mask, param=param_masking)
        
        plt.figure(figsize=(5.15, 1.99), dpi=100) # 515 x 199
        plt.imshow(time_mask, interpolation='nearest', origin='lower', aspect='auto')
        plt.axis('off')
        plt.savefig(mels_path + str(count) + ".png", dpi=100)
        plt.close()
        img = Image.open(mels_path + str(count) + ".png").convert("RGB")
        img.save(mels_path + str(count) + ".png")
        
        # Save image names with corresponding labels (0: no COVID, 1: COVID)
        labels_list.append({'filename': f"{count}.png", 'label': label})
        count+=1
    
    # Save labels
    Y = pd.DataFrame(labels_list)
    Y.to_csv(labels_path,index=False)


def MelSpecto(waves_path,meanSignalLength, labels_cut_audio_path, mels_path, melspec_labels_path):
    Y = pd.DataFrame(columns=["uuid","cut_audio", "cut_melspec", "status"])
    label_audio = pd.read_csv(labels_cut_audio_path)
    #for fn in os.listdir(waves_path):
    for fn in label_audio['non_silent_name'].values:
        if fn.endswith(".wav"):
            label = fn.split(".")[0]
            signal, sr = librosa.load(waves_path + fn)
            s_len = len(signal)
            ## Add zero padding to the signal if less than mean / Remove from begining and the end if signal length is greater than mean 
            if s_len < meanSignalLength:
                pad_len = meanSignalLength - s_len
                pad_rem = pad_len % 2
                pad_len //= 2
                signal = np.pad(
                    signal, (pad_len, pad_len + pad_rem), "constant", constant_values=0
                )
            else:
                pad_len = s_len - meanSignalLength
                pad_len //= 2
                signal = signal[pad_len : pad_len + meanSignalLength]
            mel_spectrogram = librosa.feature.melspectrogram(
                y=signal,
                sr=sr,
                n_mels=128,
                hop_length=512,
                fmax=8000,
                n_fft=512,
                center=True,
            )
            dbscale_mel_spectrogram = librosa.power_to_db(
                mel_spectrogram, ref=np.max, top_db=80
            )
            plt.figure(figsize=(5.15, 1.99), dpi=100)  # Dimensions: 515 x 199
            plt.imshow(dbscale_mel_spectrogram, interpolation='nearest', origin='lower', aspect='auto')
            plt.axis('off')
            plt.savefig(mels_path + label+ ".png", dpi=100)
            plt.close()
            img = Image.open(mels_path + label+ ".png").convert("RGB")
            img.save(mels_path + label+ ".png")
            Y = pd.concat([Y, pd.DataFrame([{"uuid": label_audio[label_audio['non_silent_name']==fn]['uuid'].values[0], "cut_audio": fn, "cut_melspec":label + ".png","status": label_audio[label_audio['non_silent_name']==fn]['label'].values[0]}])], ignore_index=True)
    Y.to_csv(melspec_labels_path, index=False)
