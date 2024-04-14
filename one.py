import librosa
import numpy as np
import os
import random
import soundfile as sf

BPM = 73.96
SECONDS_PER_BAR = 60 / BPM * 4  # Assuming 4 beats per bar
TARGET_SAMPLE_RATE = 44100  # Target sample rate
TOTAL_BARS = 41  # Total bars in the original song
REUSE_DELAY_SECONDS = 15 * 60  # 10 minutes converted to seconds

# Load an audio file with librosa
def load_audio(file_path, sr=TARGET_SAMPLE_RATE):
    data, _ = librosa.load(file_path, sr=sr, mono=True)  # Ensure mono is True for consistent array shapes
    return data

# Load the original track
original_data = load_audio('original.wav')

# Discover and load slices
slices = {}
for file in os.listdir('.'):
    if file.startswith('slice') and file.endswith('.wav'):
        parts = file.rstrip('.wav').split('_bar')
        start_bar, end_bar = int(parts[1]), int(parts[2])
        slices[start_bar] = slices.get(start_bar, []) + [{'end_bar': end_bar, 'data': load_audio(file), 'filename': file}]

def assemble_audio(duration_minutes, original_data, slices):
    total_seconds = duration_minutes * 60
    current_bar = 1
    assembled_audio = np.array([], dtype=np.float32)
    slice_usage_times = {}  # Tracks when slices were last used

    while len(assembled_audio) / TARGET_SAMPLE_RATE < total_seconds:
        if current_bar in slices:
            # Filter out slices based on reuse delay
            available_slices = [s for s in slices[current_bar] if s['filename'] not in slice_usage_times or (len(assembled_audio) / TARGET_SAMPLE_RATE - slice_usage_times[s['filename']] >= REUSE_DELAY_SECONDS)]
            if available_slices and random.choice([True, True, True, True, False]):  # 50% chance to play a slice
                selected_slice = random.choice(available_slices)
                assembled_audio = np.concatenate((assembled_audio, selected_slice['data']))
                # Update the last used time for the selected slice
                slice_usage_times[selected_slice['filename']] = len(assembled_audio) / TARGET_SAMPLE_RATE
                current_bar = selected_slice['end_bar']  # Skip ahead to the end of the slice
                continue

        next_bar = current_bar + 1 if current_bar < TOTAL_BARS else 1  # Loop back to start if at end
        start_sample = int((current_bar - 1) * SECONDS_PER_BAR * TARGET_SAMPLE_RATE)
        end_sample = int((next_bar - 1) * SECONDS_PER_BAR * TARGET_SAMPLE_RATE)
        original_segment = original_data[start_sample:end_sample]
        assembled_audio = np.concatenate((assembled_audio, original_segment))
        current_bar = next_bar

    return assembled_audio

# Assemble 30 minutes of audio as an example
assembled_data = assemble_audio(30, original_data, slices)

# Save to a file
sf.write('assembled_output_30min.wav', assembled_data, TARGET_SAMPLE_RATE)
