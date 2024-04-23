import librosa
import numpy as np
import os
import random
import soundfile as sf
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)

BPM = 73.96
SECONDS_PER_BAR = 60 / BPM * 4  # Assuming 4 beats per bar
TARGET_SAMPLE_RATE = 44100  # Target sample rate
TOTAL_BARS = {'branch1': 41, 'branch2': 31, 'branch3': 27}  # Total bars in the original songs
SLICE_REUSE_DELAY_SECONDS = 18 * 60  # 18 minutes before a slice can be reused
BRANCH_REUSE_DELAY_SECONDS = 5 * 60  # 5 minutes before a non-branch1 branch can be reused
MAX_CONSECUTIVE_NON_BRANCH1_LOOPS = 1  # Maximum number of consecutive loops in non-branch1 branches

# Load an audio file with librosa
def load_audio(file_path, sr=TARGET_SAMPLE_RATE):
    data, _ = librosa.load(file_path, sr=sr, mono=True)
    return data

# Discover and load slices, initializing play counts and last played times
def discover_slices(branch_folder):
    slices = {}
    slice_metadata = {}
    for file in os.listdir(branch_folder):
        if file.startswith('slice') and file.endswith('.wav'):
            parts = file.rstrip('.wav').split('_bar')
            start_bar, end_bar = int(parts[1]), int(parts[2])
            slice_data = load_audio(os.path.join(branch_folder, file))
            slices[start_bar] = slices.get(start_bar, []) + [{'end_bar': end_bar, 'data': slice_data, 'filename': file}]
            slice_metadata[file] = {'play_count': 0, 'last_played': None}
    return slices, slice_metadata

def get_available_slices(current_bar, current_time, slices, slice_metadata):
    available_slices = [
        s for s in slices.get(current_bar, [])
        if slice_metadata[s['filename']]['last_played'] is None or
        current_time - slice_metadata[s['filename']]['last_played'] >= SLICE_REUSE_DELAY_SECONDS
    ]
    available_slices.sort(key=lambda s: slice_metadata[s['filename']]['play_count'])
    return available_slices

def assemble_audio(duration_minutes, original_data, slices, slice_metadata):
    total_seconds = duration_minutes * 60
    current_bar = 1
    current_branch = 'branch1'
    assembled_audio = np.array([], dtype=np.float32)
    non_branch1_loop_count = 0
    branch_last_played = {'branch2': None, 'branch3': None}
    branch_play_order = []

    while len(assembled_audio) / TARGET_SAMPLE_RATE < total_seconds:
        available_slices = get_available_slices(current_bar, len(assembled_audio) / TARGET_SAMPLE_RATE, slices[current_branch], slice_metadata[current_branch])
        if available_slices and random.choice([True, True, True, True, False]):
            selected_slice = random.choice(available_slices)
            assembled_audio = np.concatenate((assembled_audio, selected_slice['data']))
            slice_metadata[current_branch][selected_slice['filename']]['play_count'] += 1
            slice_metadata[current_branch][selected_slice['filename']]['last_played'] = len(assembled_audio) / TARGET_SAMPLE_RATE
            current_bar = selected_slice['end_bar']
            logging.info(f'Playing slice {selected_slice["filename"]} from {current_branch}')

            if current_branch != 'branch1' and selected_slice['end_bar'] == TOTAL_BARS[current_branch]:
                non_branch1_loop_count += 1
        else:
            next_bar = current_bar + 1
            if next_bar > TOTAL_BARS[current_branch]:
                if current_branch != 'branch1':
                    non_branch1_loop_count += 1
                    if non_branch1_loop_count >= MAX_CONSECUTIVE_NON_BRANCH1_LOOPS or \
                    (branch_last_played['branch2'] is not None and branch_last_played['branch3'] is not None):
                        current_branch = 'branch1'
                        next_bar = 13 if current_branch == 'branch2' else 1
                        non_branch1_loop_count = 0
                    else:
                        next_bar = 1
                else:
                    next_bar = 1
                    non_branch1_loop_count = 0

            start_sample = int((current_bar - 1) * SECONDS_PER_BAR * TARGET_SAMPLE_RATE)
            end_sample = int((next_bar - 1) * SECONDS_PER_BAR * TARGET_SAMPLE_RATE)
            original_segment = original_data[current_branch][start_sample:end_sample]
            assembled_audio = np.concatenate((assembled_audio, original_segment))
            current_bar = next_bar
            logging.info(f'Playing original segment from {current_branch} bars {current_bar} to {next_bar}')

        if current_branch == 'branch1' and current_bar in [13, 28]:
            next_branch = 'branch2' if current_bar == 13 else 'branch3'
            current_time = len(assembled_audio) / TARGET_SAMPLE_RATE
            if branch_last_played[next_branch] is None or current_time - branch_last_played[next_branch] >= BRANCH_REUSE_DELAY_SECONDS:
                current_branch = next_branch
                current_bar = 1
                non_branch1_loop_count = 0
                branch_last_played[next_branch] = current_time
                branch_play_order.append(current_branch)
                logging.info(f'Switching to {current_branch}')

    logging.info(f'Branch play order: {branch_play_order}')
    return assembled_audio

# Load the original tracks
original_data = {
    'branch1': load_audio('branch1/original.wav'),
    'branch2': load_audio('branch2/original.wav'),
    'branch3': load_audio('branch3/original.wav')
}

# Discover and load slices for each branch
slices = {}
slice_metadata = {}
for branch in ['branch1', 'branch2', 'branch3']:
    branch_slices, branch_metadata = discover_slices(branch)
    slices[branch] = branch_slices
    slice_metadata[branch] = branch_metadata

# Assemble 60 minutes of audio as an example
assembled_data = assemble_audio(60, original_data, slices, slice_metadata)

# Save to a file
sf.write('assembled_output.wav', assembled_data, TARGET_SAMPLE_RATE)
