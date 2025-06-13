import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim 
import whisper 
import numpy as np 
import os 
import json 
from pathlib import Path 
from tqdm import tqdm 
import evaluate 
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from dataset.torgo_dataset import WhisperDataCollator, TorgoDataset


class TrainingConfig:
    model_name = 'small.en'
    batch_size = 64
    learning_rate = 1e-5
    num_epochs = 10
    max_length = 448
    gradient_accumulation_steps = 4
    warmup_steps= 500
    save_steps = 1000
    eval_steps = 500
    output_dir = './whisper-torgo-finetuned'
    data_dir = os.path.expanduser('~/torgo_data')
    max_audio_length = 30 

class WhisperTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = whisper.load_model(config.model_name)
        self.model.to(self.device)
        
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=False
        )
        
        self.data_collator = WhisperDataCollator(self.tokenizer)
        
        self.wer_metric = evaluate.load("wer")
        
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def create_dataloaders(self):
        train_dataset = TorgoDataset(
            self.config.data_dir, 
            self.model, 
            split="train",
            max_audio_length=self.config.max_audio_length
        )
        val_dataset = TorgoDataset(
            self.config.data_dir, 
            self.model, 
            split="val",
            max_audio_length=self.config.max_audio_length
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=2
        )
        
        return train_loader, val_loader
    
    def evaluate(self, val_loader):
        self.model.eval()
        predictions = []
        references = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                mel = batch["mel"].to(self.device)
                labels = batch["labels"].to(self.device)
                text_tokens = batch["text_tokens"].to(self.device)
                
                audio_features = self.model.encoder(mel)
                
                logits = self.model.decoder(text_tokens, audio_features)
                
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                # Shift labels for causal modeling
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_loss += loss.item()
                
                for i in range(mel.size(0)):
                    single_mel = mel[i:i+1]
                    options = whisper.DecodingOptions(
                        language="en",
                        without_timestamps=True,
                        beam_size=5
                    )
                    result = whisper.decode(self.model, single_mel.squeeze(0), options)
                    predictions.append(result.text.strip())
                
                references.extend(batch["transcripts"])
        
        # Calculate WER
        wer = self.wer_metric.compute(predictions=predictions, references=references)
        avg_loss = total_loss / len(val_loader)
        
        return {
            "eval_loss": avg_loss,
            "eval_wer": wer,
            "predictions": predictions[:5],  # Save first 5 predictions for inspection
            "references": references[:5]
        }
    
    def train(self):
        train_loader, val_loader = self.create_dataloaders()
        
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1  # 10% warmup
        )
        
        # Training loop
        global_step = 0
        best_wer = float('inf')
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                try:
                    mel = batch["mel"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    text_tokens = batch["text_tokens"].to(self.device)
                    
                    # previously used: 
                    #if torch.any(text_tokens >= vocab_size) or torch.any(text_tokens < -100):
                        #    text_tokens = torch.clamp(text_tokens, 0, vocab_size - 1)
                    # however, this might have converted special tokens (like eot, sot to vocab tokens, which breaks the loop
                    vocab_size = self.tokenizer.encoding.n_vocab
                    valid_mask = (text_tokens >= -100) & (text_tokens < vocab_size)
                    
                    if not torch.all(valid_mask):
                        print(f"Warning: Skipping batch {step} due to invalid tokens")
                        print(f"Token range: {text_tokens.min().item()} to {text_tokens.max().item()}")
                        print(f"Vocab size: {vocab_size}")
                        continue  # Skip this batch instead of crashing
                    
                    audio_features = self.model.encoder(mel)
                    
                    logits = self.model.decoder(text_tokens, audio_features)
                    
                    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                    # Shift for causal language modeling
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = loss / self.config.gradient_accumulation_steps
                    
                    loss.backward()
                    total_loss += loss.item()
                    
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                    
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
                    
                    # Evaluation
                    if global_step > 0 and global_step % self.config.eval_steps == 0:
                        eval_results = self.evaluate(val_loader)
                        print(f"\nStep {global_step}:")
                        print(f"Eval Loss: {eval_results['eval_loss']:.4f}")
                        print(f"Eval WER: {eval_results['eval_wer']:.4f}")
                        print(f"Sample predictions: {eval_results['predictions'][:2]}")
                        print(f"Sample references: {eval_results['references'][:2]}")
                        
                        if eval_results['eval_wer'] < best_wer:
                            best_wer = eval_results['eval_wer']
                            self.save_model(f"best_model_wer_{best_wer:.4f}")
                            print(f"New best WER: {best_wer:.4f}")
                        
                        self.model.train()
                    
                    if global_step > 0 and global_step % self.config.save_steps == 0:
                        self.save_model(f"checkpoint-{global_step}")

                # error batch handling
                except Exception as e:
                    print(f"Error in batch {step}: {e}")
                    print(f"mel shape: {mel.shape if 'mel' in locals() else 'N/A'}")
                    print(f"text_tokens shape: {text_tokens.shape if 'text_tokens' in locals() else 'N/A'}")
                    print(f"Continuing to next batch...")
                    continue  
            
            avg_train_loss = total_loss / max(len(train_loader), 1)
            print(f"Epoch {epoch+1} - Average training loss: {avg_train_loss:.4f}")
        
        final_eval = self.evaluate(val_loader)
        print(f"\nFinal Results:")
        print(f"Final WER: {final_eval['eval_wer']:.4f}")
        
        self.save_model("final_model")
        
        return final_eval
    
    def save_model(self, checkpoint_name):
        save_path = Path(self.config.output_dir) / checkpoint_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_path = save_path / "model.pt"
        torch.save(self.model.state_dict(), model_path)
        
        config_path = save_path / "config.json"
        model_config = {
            "model_name": self.config.model_name,
            "dims": {
                "n_mels": self.model.dims.n_mels,
                "n_audio_ctx": self.model.dims.n_audio_ctx,
                "n_audio_state": self.model.dims.n_audio_state,
                "n_audio_head": self.model.dims.n_audio_head,
                "n_audio_layer": self.model.dims.n_audio_layer,
                "n_vocab": self.model.dims.n_vocab,
                "n_text_ctx": self.model.dims.n_text_ctx,
                "n_text_state": self.model.dims.n_text_state,
                "n_text_head": self.model.dims.n_text_head,
                "n_text_layer": self.model.dims.n_text_layer,
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f"Model saved to {save_path}")


