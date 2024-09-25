import os
import warnings
import matplotlib.pyplot as plt
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.module')

# Initializing the emotion recognition pipeline
emotion_pipeline = pipeline(
    task=Tasks.emotion_recognition,
    model="iic/emotion2vec_plus_large"
)

# Set up the test audio folder and the output folder
test_audios = './test_dataset/SAVEE'
output_folder = './emotion_outputs'
os.makedirs(output_folder, exist_ok=True)

# Define the mood that corresponds to the mood label folder name
emotion_map = {
    'angry': 'angry',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad'
}

# Label mapping, used to convert Chinese labels to English
label_map = {
    '生气/angry': 'angry',
    '开心/happy': 'happy',
    '中立/neutral': 'neutral',
    '难过/sad': 'sad'
}

# Record true labels and predicted labels for each emotion
true_labels_dict = {emotion: [] for emotion in emotion_map}
pred_labels_dict = {emotion: [] for emotion in emotion_map}

# Walk through each mood folder
for emotion, emotion_label in emotion_map.items():
    emotion_folder = os.path.join(test_audios, emotion)

    if not os.path.exists(emotion_folder):
        print(f"Folder {emotion_folder} does not exist. Skipping.")
        continue

    # Walk through all the audio files in the mood folder
    for audio_file in os.listdir(emotion_folder):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(emotion_folder, audio_file)

            # Emotion recognition model is used for emotion recognition
            rec_result = emotion_pipeline(audio_path, output_dir=output_folder, granularity="utterance",
                                          extract_embedding=False)

            # Get the best mood label predicted by the model
            best_label_index = rec_result[0]['scores'].index(max(rec_result[0]['scores']))
            best_label = rec_result[0]['labels'][best_label_index]

            true_labels_dict[emotion].append(emotion_label)
            pred_labels_dict[emotion].append(label_map.get(best_label, best_label))
            print(f"Results for {audio_file}: Predicted: {best_label}, Actual: {emotion_label}")

# Calculate the accuracy of each emotion
accuracy_dict = {}
for emotion in emotion_map:
    accuracy = accuracy_score(true_labels_dict[emotion], pred_labels_dict[emotion])
    accuracy_dict[emotion] = accuracy
    print(f"Accuracy for {emotion}: {accuracy:.4f}")

# Overall accuracy of computation
true_labels = sum(true_labels_dict.values(), [])
pred_labels = sum(pred_labels_dict.values(), [])
overall_accuracy = accuracy_score(true_labels, pred_labels)
print(f"Overall Accuracy: {overall_accuracy:.4f}")

# Use matplotlib to generate a bar chart
emotions = list(accuracy_dict.keys())
accuracies = list(accuracy_dict.values())
accuracies.append(overall_accuracy)
emotions.append('Overall')

plt.figure(figsize=(10, 6))
plt.bar(emotions, accuracies, color='skyblue')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Emotion Recognition Accuracy by Category')
plt.savefig('emotion_accuracy.png')
