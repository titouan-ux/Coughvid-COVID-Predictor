import numpy as np
import pandas as pd
import os
import subprocess
from pathlib import Path


def convert_files(folder):
    """Convert files from .webm and .ogg to .wav
    folder: path to coughvid database and metadata_compiled csv"""

    df = pd.read_csv(folder + "metadata_compiled.csv")
    names_to_convert = df.uuid.to_numpy()
    for counter, name in enumerate(names_to_convert):
        if counter % 1000 == 0:
            print("Finished {0}/{1}".format(counter, len(names_to_convert)))
        if os.path.isfile(folder + name + ".webm"):
            subprocess.call(
                ["ffmpeg", "-i", folder + name + ".webm", folder + name + ".wav"]
            )
        elif os.path.isfile(folder + name + ".ogg"):
            subprocess.call(
                ["ffmpeg", "-i", folder + name + ".ogg", folder + name + ".wav"]
            )
        else:
            print("Error: No file name {0}".format(name))


def convert_files_updated(folder, df):
    """Convert .webm/.ogg to proper PCM .wav (16-bit, 16kHz) and delete source if successful."""

    names_to_convert = df.uuid.to_numpy()
    for counter, name in enumerate(names_to_convert):
        if counter % 1000 == 0:
            print(f"Finished {counter}/{len(names_to_convert)}")

        wav_file = os.path.join(folder, name + ".wav")
        webm_file = os.path.join(folder, name + ".webm")
        ogg_file = os.path.join(folder, name + ".ogg")

        # If .wav already exists, clean up old sources
        if os.path.isfile(wav_file):
            if os.path.isfile(webm_file):
                try:
                    os.remove(webm_file)
                    print(f"Deleted existing .webm file: {name}.webm")
                except Exception as e:
                    print(f"Could not delete {name}.webm: {e}")
            elif os.path.isfile(ogg_file):
                try:
                    os.remove(ogg_file)
                    print(f"Deleted existing .ogg file: {name}.ogg")
                except Exception as e:
                    print(f"Could not delete {name}.ogg: {e}")
            continue

        # Pick source file
        input_file = None
        if os.path.isfile(webm_file):
            input_file = webm_file
        elif os.path.isfile(ogg_file):
            input_file = ogg_file

        if input_file:
            result = subprocess.call([
                "ffmpeg", "-y", "-i", input_file,
                "-acodec", "pcm_s16le", "-ar", "16000", wav_file
            ])
            if result == 0 and os.path.isfile(wav_file):
                try:
                    os.remove(input_file)
                    print(f"Converted and deleted: {os.path.basename(input_file)}")
                except Exception as e:
                    print(f"Could not delete {input_file}: {e}")
            else:
                print(f"Error converting: {input_file}")
        else:
            print(f"Error: No source file for {name}")

import os
import subprocess
import pandas as pd
import wave

def is_valid_wav(filepath):
    """Check if WAV file is a valid PCM file with RIFF header."""
    try:
        with wave.open(filepath, 'rb') as wf:
            return wf.getcomptype() == 'NONE'  # 'NONE' means uncompressed PCM
    except wave.Error:
        return False
    except:
        return False

def convert_to_valid_wav(input_path, output_path):
    """Use ffmpeg to convert to proper PCM WAV format (16-bit, 16kHz)."""
    subprocess.call([
        "ffmpeg", "-y", "-i", input_path,
        "-acodec", "pcm_s16le", "-ar", "16000", output_path
    ])

def process_files(output_folder, origin_folder, df):
    uuids = df.uuid.to_numpy()

    for i, uuid in enumerate(uuids):
        wav_path = os.path.join(output_folder, uuid + ".wav")

        # If valid WAV already exists, skip
        if os.path.exists(wav_path) and is_valid_wav(wav_path):
            continue

        # Try to find and convert original source
        ogg_path = os.path.join(origin_folder, uuid + ".ogg")
        webm_path = os.path.join(origin_folder, uuid + ".webm")
        

        if os.path.exists(ogg_path):
            print(f"[{i}] Re-converting from .ogg: {uuid}")
            convert_to_valid_wav(ogg_path, wav_path)
        elif os.path.exists(webm_path):
            print(f"[{i}] Re-converting from .webm: {uuid}")
            convert_to_valid_wav(webm_path, wav_path)
        else:
            print(f"[{i}] Missing source for: {uuid}")


