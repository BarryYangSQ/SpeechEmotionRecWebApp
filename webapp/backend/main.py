from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import torch
import shutil
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

app = Flask(__name__)
CORS(app)

# Configure upload folder and results folder
UPLOAD_FOLDER = './uploads'
RESULTS_FOLDER = './results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

# Initialize models
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
(get_speech_ts, save_audio, read_audio, VADIterator, collect_chunks) = utils

asr_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='iic/SenseVoiceSmall',
    model_revision="master",
)

emotion_pipeline = pipeline(
    task=Tasks.emotion_recognition,
    model="iic/emotion2vec_plus_large",
)

def extract_segment(audio, start_time, end_time, sample_rate):
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    segment = audio[start_sample:end_sample]
    return segment

def process_audio_file(audio_file):
    results = []

    # Perform emotion recognition on the entire audio file
    overall_emotion_result = emotion_pipeline(audio_file, granularity="utterance", extract_embedding=False)
    best_overall_label_index = overall_emotion_result[0]['scores'].index(max(overall_emotion_result[0]['scores']))
    best_overall_emotion = overall_emotion_result[0]['labels'][best_overall_label_index]

    # Add the overall emotion result as the first entry in the JSON result
    results.append({
        "Overall Emotion": best_overall_emotion
    })

    # Perform VAD segmentation and recognition on each segment
    audio = read_audio(audio_file)
    vad_segments = get_speech_ts(audio, vad_model)

    for i, segment in enumerate(vad_segments):
        start_time = segment['start'] / 16000
        end_time = segment['end'] / 16000
        segment_audio = extract_segment(audio, start_time, end_time, 16000)
        segment_path = os.path.join(RESULTS_FOLDER,
                                    f"{os.path.splitext(os.path.basename(audio_file))[0]}_segment_{i + 1}_{start_time:.2f}-{end_time:.2f}.wav")
        save_audio(segment_path, segment_audio, 16000)

        asr_result = asr_pipeline(segment_path)
        if isinstance(asr_result, list) and len(asr_result) > 0:
            result = asr_result[0]
            text_content = result.get("text", "")

            rec_result = emotion_pipeline(segment_path, granularity="utterance", extract_embedding=False)
            best_label_index = rec_result[0]['scores'].index(max(rec_result[0]['scores']))
            best_emotion = rec_result[0]['labels'][best_label_index]

            parts = text_content.split('<|')
            extracted_info = []
            for part in parts:
                if '|>' in part:
                    extracted_info.append(part.split('|>')[0].strip())

            if len(extracted_info) >= 4:
                language = extracted_info[0]
                emotion = best_emotion
                audio_type = extracted_info[2]
                with_or_wo_itn = extracted_info[3]
                text = text_content.split('|>')[-1].strip()

                result_dict = {
                    "Language": language,
                    "Emotion": emotion,
                    "Audio Type": audio_type,
                    "Text": text
                }
                results.append(result_dict)

    result_file_path = os.path.join(RESULTS_FOLDER, 'output_results.json')
    with open(result_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # Remove sub-audio files from the results folder
    for file in os.listdir(RESULTS_FOLDER):
        if file.endswith('.wav') and file != 'output_results.json':
            os.remove(os.path.join(RESULTS_FOLDER, file))

@app.route('/upload', methods=['POST'])
def upload_file():
    # Clear the upload folder
    for file in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return jsonify({"message": "File uploaded successfully", "filename": file.filename}), 200

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    response.headers['Cache-Control'] = 'public, max-age=3600'
    return response

@app.route('/recognize', methods=['POST'])
def recognize_file():
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({"error": "Filename not provided"}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    process_audio_file(filepath)
    return jsonify({"message": "Recognition completed"}), 200

@app.route('/results', methods=['GET'])
def get_results():
    results_file_path = os.path.join(RESULTS_FOLDER, 'output_results.json')
    
    try:
        with open(results_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

if __name__ == '__main__':
    app.run(debug=True)
