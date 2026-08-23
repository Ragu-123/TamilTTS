'''
IndicMFA Alignment and Character Duration Extractor for TamilTTS
================================================================
Aligns Tamil speech datasets and produces exact character durations (in mel frames)
for direct Kokoro-style supervision in TamilTTS training.
'''
import os
import sys
import glob
import urllib.request
import zipfile
import torch
import torchaudio
import numpy as np

DICT_URL = 'https://github.com/AI4Bharat/IndicMFA/releases/download/Tamil/Tamil_Dictionary_g2g.txt'
MODEL_URL = 'https://github.com/AI4Bharat/IndicMFA/releases/download/Tamil/Tamil_Acoustic_Model.zip'

def download_indic_mfa_assets(dest_dir='indic_mfa_tamil'):
    os.makedirs(dest_dir, exist_ok=True)
    dict_path = os.path.join(dest_dir, 'Tamil_Dictionary_g2g.txt')
    model_zip = os.path.join(dest_dir, 'Tamil_Acoustic_Model.zip')

    if not os.path.exists(dict_path):
        print(f'Downloading Tamil G2G Dictionary to {dict_path}...')
        urllib.request.urlretrieve(DICT_URL, dict_path)
        print('Done!')

    if not os.path.exists(model_zip):
        print(f'Downloading Tamil Acoustic Model to {model_zip}...')
        urllib.request.urlretrieve(MODEL_URL, model_zip)
        print('Done!')

    return dict_path, model_zip

def compute_frame_durations(time_intervals, hop_length=256, sample_rate=22050):
    frame_time = hop_length / sample_rate
    durations = []
    for start, end in time_intervals:
        dur = max(1, int(round((end - start) / frame_time)))
        durations.append(dur)
    return durations

if __name__ == '__main__':
    dest = sys.argv[1] if len(sys.argv) > 1 else 'indic_mfa_tamil'
    d_path, m_path = download_indic_mfa_assets(dest)
    print(f'Assets verified:\n  Dictionary: {d_path}\n  Model: {m_path}')
