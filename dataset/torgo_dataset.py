import os
import torch
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio
import whisper
import pickle
import numpy as np
from tqdm import tqdm
import re


class TorgoDataset(Dataset):
    def __init__(self, data_root, processor, split='train', max_audio_length=30, 
                 cache_dir=None, preprocess=True):
        self.data_root = Path(data_root)
        self.processor = processor
        self.max_audio_length = max_audio_length  
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir else Path(data_root) / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
        
        self.cache_file = self.cache_dir / f"{split}_preprocessed.pkl"
        
        if preprocess and self.cache_file.exists():
            print(f"Loading preprocessed data from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.preprocessed_data = pickle.load(f)
        elif preprocess:
            print("Preprocessing data...")
            self.samples = self._load_data()
            self.preprocessed_data = self._preprocess_all_data()
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.preprocessed_data, f)
            print(f"Saved preprocessed data to {self.cache_file}")
        else:
            self.samples = self._load_data()
            self.preprocessed_data = None
    
    def _is_valid_transcription(self, transcription):
        """Check if transcription is valid and clean it if possible"""
        original_transcription = transcription
        
        if transcription.startswith('['):
            return None, f"starts with square bracket: {transcription[:50]}"
        
        if ('/' in transcription or 
            transcription.lower().endswith(('.jpg', '.png', '.gif', '.bmp', '.jpeg', '.wav', '.mp3', '.mp4', '.avi'))):
            return None, f"contains file path: {transcription[:50]}"
        
        if re.search(r'x{3,}', transcription, re.IGNORECASE):
            return None, f"contains multiple x's: {transcription[:50]}"
        
        if '[' in transcription and ']' in transcription:
            # if doesnt start w/ square brackets, clear brackets and its contents and remove whitespace
            cleaned = re.sub(r'\[.*?\]', '', transcription)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            transcription = cleaned
            print(f"Cleaned transcription: '{original_transcription}' -> '{transcription}'")
        
        transcription = re.sub(r'[^\w\s]', '', transcription)
        
        # Ensure transcription is not empty after cleaning
        if not transcription or not transcription.strip():
            return None, f"empty after cleaning: {original_transcription[:50]}"
        
        # Limit transcription length
        transcription = transcription[:448]
        
        # Ensure transcription is valid UTF-8
        transcription = transcription.encode('utf-8', 'ignore').decode('utf-8')
        
        return transcription, None
        
    def _load_data(self):
        if self.split == 'train':
            speakers = ['F01', 'F04', 'M01', 'M05', 'FC01', 'FC03', 'MC01', 'MC03', 'MC04', 'M04']
        elif self.split == 'val':
            speakers = ['M03', 'FC02', 'MC02']
        elif self.split == 'test':
            speakers = ['F03', 'M02']
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
        samples = []
        skipped_count = 0
        skip_reasons = {}
    
        print(f"Loading data from: {self.data_root}, split: {self.split}")
    
        for speaker in speakers:
            speaker_dir = self.data_root / speaker
            if not speaker_dir.exists():
                print(f"Warning: Speaker directory {speaker_dir} does not exist")
                continue
    
            session_dirs = [d for d in speaker_dir.iterdir() if d.is_dir() and d.name.startswith('Session')]
            print(f"Found {len(session_dirs)} sessions for speaker {speaker}")
    
            for session in session_dirs:
                wav_dirs = [d for d in session.iterdir() if d.is_dir() and d.name.startswith('wav_')]
                if not wav_dirs:
                    print(f"No wav_ directories found in {session}")
                    continue
    
                wav_dir = wav_dirs[0]  # pick first available mic
                prompt_dir = session / 'prompts'
    
                if not prompt_dir.exists():
                    print(f"Missing prompts directory in {session}")
                    continue
    
                wav_files = list(wav_dir.glob('*.wav'))
                print(f"Found {len(wav_files)} wav files in {wav_dir}")
    
                for wav_file in wav_files:
                    utt_id = wav_file.stem
                    prompt_file = prompt_dir / f'{utt_id}.txt'
    
                    if not prompt_file.exists():
                        print(f"Missing prompt file: {prompt_file}")
                        continue
    
                    try:
                        with open(prompt_file, 'r') as f:
                            raw_transcription = f.read().strip()
                        
                        # Validate and clean transcription
                        transcription, skip_reason = self._is_valid_transcription(raw_transcription)
                        
                        if transcription is None:
                            skipped_count += 1
                            skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
                            continue
                            
                    except Exception as e:
                        print(f"Error reading {prompt_file}: {e}")
                        continue
    
                    samples.append({
                        'audio_path': wav_file,
                        'transcription': transcription,
                        'speaker': speaker
                    })
    
        print(f"Loaded {len(samples)} samples for {self.split} split")
        print(f"Skipped {skipped_count} samples due to invalid transcriptions:")
        for reason, count in skip_reasons.items():
            print(f"  - {reason}: {count} samples")
        
        return samples

    def _preprocess_single_sample(self, sample):
        """Preprocess a single sample - extract mel and tokenize text"""
        audio_path = sample['audio_path']
        transcription = sample['transcription']
        
        try:
            audio = whisper.load_audio(str(audio_path))
            audio = whisper.pad_or_trim(audio)
            
            mel = whisper.log_mel_spectrogram(audio) 
            
            text_tokens = self._tokenize_text(transcription)
            
            return {
                'mel': mel,
                'text_tokens': text_tokens,
                'transcription': transcription,
                'speaker': sample['speaker']
            }
        except Exception as e:
            print(f"Error preprocessing {audio_path}: {e}")
            return None
    
    def _preprocess_all_data(self):
        """Preprocess all samples and return list of preprocessed data"""
        preprocessed = []
        
        for sample in tqdm(self.samples, desc="Preprocessing samples"):
            processed = self._preprocess_single_sample(sample)
            if processed is not None:
                preprocessed.append(processed)
        
        return preprocessed
    
    def _tokenize_text(self, transcription):
        """Tokenize text with proper validation"""
        text_tokens = []
        
        text_tokens.append(self.tokenizer.sot)
        
        lang_token = self.tokenizer.sot + 1 + self.tokenizer.encoding.encode("en")[0]
        text_tokens.append(lang_token)
        
        text_tokens.append(self.tokenizer.transcribe)
        
        text_tokens.append(self.tokenizer.no_timestamps)
        
        try:
            encoded_text = self.tokenizer.encoding.encode(transcription)
            valid_tokens = [t for t in encoded_text if 0 <= t < self.tokenizer.encoding.n_vocab]
            text_tokens.extend(valid_tokens)
        except Exception as e:
            print(f"Error encoding transcription '{transcription}': {e}")
            text_tokens.extend([])
        
        text_tokens.append(self.tokenizer.eot)
        
        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        
        vocab_size = self.tokenizer.encoding.n_vocab
        if torch.any(text_tokens >= vocab_size) or torch.any(text_tokens < 0):
            print(f"Warning: Invalid tokens found")
            text_tokens = torch.clamp(text_tokens, 0, vocab_size - 1)

        if len(text_tokens) > 448:
            text_tokens = text_tokens[:448]
            text_tokens[-1] = self.tokenizer.eot

        return text_tokens

    def __len__(self):
        if self.preprocessed_data is not None:
            return len(self.preprocessed_data)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.preprocessed_data is not None:
            sample = self.preprocessed_data[idx]
            return {
                'mel': sample['mel'].unsqueeze(0),  # add batch dimension 
                'text_tokens': sample['text_tokens'],
                'transcription': sample['transcription'],
                'speaker': sample['speaker']
            }
        else:
            return self._get_item_slow(idx)
    
    def _get_item_slow(self, idx):
        # original slow implementation, this method causes the GPU to idle alot during training
        sample = self.samples[idx]
        audio_path = sample['audio_path']
        transcription = sample['transcription']

        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).unsqueeze(0)

        text_tokens = self._tokenize_text(transcription)

        return {
            'mel': mel,
            'text_tokens': text_tokens, 
            'transcription': transcription,
            'speaker': sample["speaker"]
        }


class WhisperDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        mels = [f["mel"] for f in features]
        mel_batch = torch.stack(mels)  # Shape: (batch_size, 1, n_mels, time_steps)
        mel_batch = mel_batch.squeeze(1)  # Shape: (batch_size, n_mels, time_steps)
        
        max_len = max(len(f["text_tokens"]) for f in features)
        vocab_size = self.tokenizer.encoding.n_vocab
        
        text_tokens = []
        labels = []
        
        for f in features:
            tokens = f["text_tokens"]
            
            if torch.any(tokens >= vocab_size) or torch.any(tokens < 0):
                print(f"Warning: Invalid tokens in batch, clamping to valid range")
                tokens = torch.clamp(tokens, 0, vocab_size - 1)
            
            padded = torch.full((max_len,), self.tokenizer.eot, dtype=torch.long)
            padded[:len(tokens)] = tokens
            
            label = padded.clone()
            label[:4] = -100  
            label[len(tokens):] = -100  
            
            text_tokens.append(padded)
            labels.append(label)
        
        return {
            "mel": mel_batch,
            "text_tokens": torch.stack(text_tokens),
            "labels": torch.stack(labels),
            "transcripts": [f["transcription"] for f in features]
        }