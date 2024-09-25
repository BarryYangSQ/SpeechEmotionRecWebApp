import os
import torch
import json
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import tkinter as tk
from tkinter import filedialog

# Initialize the Silero VAD
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
(get_speech_ts, save_audio, read_audio, VADIterator, collect_chunks) = utils

# Initializing the SenseVoice pipeline
asr_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='iic/SenseVoiceSmall',
    model_revision="master",
)

# Initializing the emotion2vec pipeline
emotion_pipeline = pipeline(
    task=Tasks.emotion_recognition,
    model="iic/emotion2vec_plus_large",
)

# Set up the output folder and document
vad_output_folder = './vad_outputs'
os.makedirs(vad_output_folder, exist_ok=True)
output_file = 'output_results.json'


def extract_segment(audio, start_time, end_time, sample_rate):
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    segment = audio[start_sample:end_sample]
    return segment


def process_audio_file(audio_file):
    results = []
    audio_path = audio_file
    # Use Silero VAD for audio processing
    audio = read_audio(audio_path)
    vad_segments = get_speech_ts(audio, vad_model)

    # Extract and save speech snippets based on VAD results
    for i, segment in enumerate(vad_segments):
        start_time = segment['start'] / 16000  # 16,000 refers to the sample rate
        end_time = segment['end'] / 16000
        segment_audio = extract_segment(audio, start_time, end_time, 16000)
        segment_path = os.path.join(vad_output_folder,
          f"{os.path.splitext(os.path.basename(audio_file))[0]}_segment_{i + 1}_{start_time:.2f}-{end_time:.2f}.wav")
        save_audio(segment_path, segment_audio, 16000)

        # Use SenseVoice to convert speech to text
        asr_result = asr_pipeline(segment_path)
        if isinstance(asr_result, list) and len(asr_result) > 0:
            result = asr_result[0]
            text_content = result.get("text", "")

            # use emotion2vec to recognize emotion
            rec_result = emotion_pipeline(segment_path, granularity="utterance", extract_embedding=False)

            # Find the best emotional label
            best_label_index = rec_result[0]['scores'].index(max(rec_result[0]['scores']))
            best_emotion = rec_result[0]['labels'][best_label_index]

            # Use <| and |> as delimiters
            parts = text_content.split('<|')
            extracted_info = []
            for part in parts:
                if '|>' in part:
                    extracted_info.append(part.split('|>')[0].strip())

            if len(extracted_info) >= 4:
                language = extracted_info[0]  # Extract language
                emotion = best_emotion  # Use the result from emotion2vec
                audio_type = extracted_info[2]  # Extract speech type
                # with_or_wo_itn = extracted_info[3]  # Extract with_or_wo_itn

                text = text_content.split('|>')[-1].strip()  # Extract text

                # Add the result to the list
                result_dict = {
                    "Language": language,
                    "Emotion": emotion,
                    "Audio Type": audio_type,
                    "Text": text
                }
                results.append(result_dict)
            else:
                print(f"Unexpected format for text field in {audio_file}: {text_content}")
        else:
            print(f"Failed to transcribe {segment_path}")

    # Write the results to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def select_file_and_process():
    root = tk.Tk()
    root.withdraw()  # Hide main window
    file_path = filedialog.askopenfilename(title="Select audio file", filetypes=[("WAV files", "*.wav")])
    if file_path:
        process_audio_file(file_path)


# Start file selection and processing
select_file_and_process()
