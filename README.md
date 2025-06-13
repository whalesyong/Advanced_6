# BBC 2025 TEAM A6

### Quick Start 
1. Download the model and the configuration file at the link: https://drive.google.com/drive/folders/1AkhAJPA3ZA5AoI9QQvo6m7RC9Zv0lhH3?usp=sharing 

2. The `model_test.ipynb` notebook has some code to load the model and its config file. In essence, we can use the Whisper API to load the fine-tuned model, provided we have the `decode_options`. More information can be found in the `model_loader.py` file. 

3. To run fine-tuning, download the dataset from the torgo database https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html, and then simply do `python train.py`.

