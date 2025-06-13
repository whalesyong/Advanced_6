from trainer import TrainingConfig, WhisperTrainer


config = TrainingConfig
trainer = WhisperTrainer(config) 

results = trainer.train()
