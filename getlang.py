import whisper
import argparse

class Getlang:
    def __init__(self):
        self.model = whisper.load_model("small")

    def detect(self, file_path):
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        return max(probs, key=probs.get)

if __name__ == "__main__":
    # Create a parser for the command line arguments
    parser = argparse.ArgumentParser(description="Detect the language spoken in an audio file.")
    parser.add_argument("file_path", help="The path to the audio file to process")

    # Parse the command line arguments
    args = parser.parse_args()

    # Create a Whisper object and use the detect method with the command-line provided file path
    getlang = Getlang()
    try:
        result = getlang.detect(args.file_path)
        print(f"The detected language is: {result}")
    except Exception as e:
        print(f"An error occurred: {e}")
