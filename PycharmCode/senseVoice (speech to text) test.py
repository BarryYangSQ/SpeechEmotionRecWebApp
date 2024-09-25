import os
import warnings
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Ignore user warnings that do not affect project operation
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.module')

# initial SenseVoice ASR pipeline
asr_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='iic/SenseVoiceSmall',
    model_revision="master"
)

# Set up the test audio folder
test_audios = './test_audios'

# Process each audio file
for audio_file in os.listdir(test_audios):
    if audio_file.endswith('.wav'):
        audio_path = os.path.join(test_audios, audio_file)

        # Use SenseVoice to convert speech to text
        asr_result = asr_pipeline(audio_path)
        print(f"ASR Result for {audio_file}: ")
        if isinstance(asr_result, list) and len(asr_result) > 0:
            result = asr_result[0]
            text_content = result.get("text", "")

            # Use <| and |> as delimiters
            parts = text_content.split('<|')
            extracted_info = []
            for part in parts:
                if '|>' in part:
                    extracted_info.append(part.split('|>')[0].strip())

            if len(extracted_info) >= 4:
                language = extracted_info[0]  # The first element is language
                emotion = extracted_info[1]  # The second element is emotion
                audio_type = extracted_info[2]  # The third element is the audio type
                with_or_wo_itn = extracted_info[3]  # The fourth element is the audio type

                text = text_content.split('|>')[-1].strip()  # Extract the last section of text

                # Print the separate parts
                print(f"Language: {language}")
                print(f"Emotion: {emotion}")
                print(f"Audio Type: {audio_type}")
                print(f"with_or_wo_itn: {with_or_wo_itn}")
                print(f"Text: {text}")
                print()
            else:
                print(f"Unexpected format for text field in {audio_file}: {text_content}")
        else:
            print(f"Failed to transcribe {audio_file}")
