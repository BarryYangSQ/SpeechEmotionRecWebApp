import os
import warnings
import torch

# Ignore specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.module')

# Initialize Silero VAD
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
(get_speech_ts, save_audio, read_audio, VADIterator, collect_chunks) = utils

# Set up test audio folder and output folder
test_audios = './test_audios'
output_folder = './vad_outputs'
os.makedirs(output_folder, exist_ok=True)

def extract_segment(audio, start_time, end_time, sample_rate):
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    segment = audio[start_sample:end_sample]
    return segment

# Process each audio file
for audio_file in os.listdir(test_audios):
    if audio_file.endswith('.wav'):
        audio_path = os.path.join(test_audios, audio_file)

        # Process audio with Silero VAD
        audio = read_audio(audio_path)
        vad_segments = get_speech_ts(audio, vad_model)

        # Extract and save speech segments based on VAD results
        for i, segment in enumerate(vad_segments):
            start_time = segment['start'] / 16000  # Sampling rate is 16000
            end_time = segment['end'] / 16000
            segment_audio = extract_segment(audio, start_time, end_time, 16000)
            segment_path = os.path.join(output_folder, f"{os.path.splitext(audio_file)[0]}_segment_{start_time:.2f}-{end_time:.2f}.wav")
            save_audio(segment_path, segment_audio, 16000)
            print(f"Saved segment: {segment_path}")
