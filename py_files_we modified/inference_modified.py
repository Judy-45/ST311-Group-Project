#modified file to run inference on a folder of audio files
import os
import sys
import json
import numpy as np
import argparse
import librosa
import torch
#from utilities import get_filename
from models import *
from pytorch_utils import move_data_to_device
import csv
import csv as pycsv

# Load label
with open('metadata/class_labels_indices.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    lines = list(reader)

the_labels = []
ids = []    # Each label has a unique id such as "/m/068hy"
for i1 in range(1, len(lines)):
    id = lines[i1][1]
    kklabel = lines[i1][2]
    ids.append(id)
    the_labels.append(kklabel)

the_classes_num = len(the_labels)

def audio_tagging_file(model, labels, device, args, audio_path, threshold=0.3):
    waveform, _ = librosa.core.load(audio_path, sr=args.sample_rate, mono=True)
    waveform = waveform[None, :]
    waveform = move_data_to_device(waveform, device)

    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    predicted_tags = [labels[i] for i, prob in enumerate(clipwise_output) if prob >= threshold]
    return predicted_tags

def audio_tagging_folder(args):
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    Model = eval(args.model_type)
    model = Model(
        sample_rate=args.sample_rate,
        window_size=args.window_size,
        hop_size=args.hop_size,
        mel_bins=args.mel_bins,
        fmin=args.fmin,
        fmax=args.fmax,
        classes_num=the_classes_num
    )

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    if 'cuda' in str(device):
        model.to(device)
        model = torch.nn.DataParallel(model)

    labels = the_labels
    results = []

    # Run inference
    audio_files = [f for f in os.listdir(args.audio_folder) if f.endswith('.wav')]
    for fname in audio_files:
        path = os.path.join(args.audio_folder, fname)
        tags = audio_tagging_file(model, labels, device, args, path, threshold=0.3)
        results.append([fname] + tags)
        print(f"{fname}: {tags}")

    # Save to CSV
    with open("inference_results_folder_single.csv", "w", newline='', encoding='utf-8') as f:
        writer = pycsv.writer(f)
        writer.writerow(["filename"] + [f"tag{i+1}" for i in range(10)])  # header (up to 10 tags)
        for row in results:
            writer.writerow(row + [""] * (11 - len(row)))  # pad to 11 columns if needed

    print("\n All results saved to inference_results_folder_single.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch audio tagging')
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--window_size', type=int, default=512)
    parser.add_argument('--hop_size', type=int, default=160)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=8000)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--audio_folder', type=str, required=True)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()

    audio_tagging_folder(args)