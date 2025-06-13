import torch
import whisper
import json
from pathlib import Path
import numpy as np

# general loader class, put in a class in case we need other methods 
class WhisperModelLoader:
    @staticmethod
    def transcribe_with_finetuned_model(model, audio_path, language="en", **decode_options):
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)
        device = next(model.parameters()).device

        mel = whisper.log_mel_spectrogram(audio).to(device)
        
        options = whisper.DecodingOptions(
            language=language,
            without_timestamps=True,
            **decode_options
        )
        
        with torch.no_grad():
            result = whisper.decode(model, mel, options)
        
        return result


def load_model_simple(checkpoint_dir):
    
    # load the model weights from the checkpoint_dir, which shld contain the .pt , and 
    # config.json file
    checkpoint_path = Path(checkpoint_dir)
    
    with open(checkpoint_path / "config.json", 'r') as f:
        config = json.load(f)

    model = whisper.load_model(config['model_name'])
    model.load_state_dict(torch.load(checkpoint_path / "model.pt"))
    model.eval()
    
    return model
