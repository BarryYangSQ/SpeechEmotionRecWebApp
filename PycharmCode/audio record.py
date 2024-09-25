import pyaudio
import wave


# This code is used to quickly record a test audio
def record_audio(duration=30, output_filename="microphone_input.wav"):
    # Set microphone input parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    audio = pyaudio.PyAudio()

    # Open mack air flow
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []

    # Set microphone input parameters
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recording to a file
    wave_file = wave.open(output_filename, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()


if __name__ == "__main__":
    record_duration = 30  # Set the recording duration to 30 seconds
    output_file = "microphone_input.wav"
    record_audio(duration=record_duration, output_filename=output_file)
