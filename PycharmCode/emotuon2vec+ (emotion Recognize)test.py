import os
import warnings
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Ignore specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.module')

# Initialize emotion recognition pipeline
emotion_pipeline = pipeline(
    task=Tasks.emotion_recognition,
    model="iic/emotion2vec_plus_large"
)

# Set up test audio folder and output folder
test_audios = './test_audios'
output_folder = './emotion_outputs'
os.makedirs(output_folder, exist_ok=True)

# Process each audio file
for audio_file in os.listdir(test_audios):
    if audio_file.endswith('.wav'):
        audio_path = os.path.join(test_audios, audio_file)

        # Use emotion recognition model to detect emotions
        rec_result = emotion_pipeline(audio_path, output_dir=output_folder, granularity="utterance",
                                      extract_embedding=False)

        # Print results
        for result in rec_result:
            best_label_index = result['scores'].index(max(result['scores']))
            best_label = result['labels'][best_label_index]
            print(f"Results for {audio_file}: {best_label}")