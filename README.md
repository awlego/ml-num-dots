# ML Dot Counter Evaluation

This project evaluates the ability of various language models to count dots/periods in sequences, testing their basic counting capabilities and exploring how different prompting strategies and sequence formats affect accuracy.

## Overview

The project tests different language models (like deepseek-r1 variants) on their ability to count dots in sequences, with variations in:
- Model sizes (1.5B, 7B, 14B, 32B parameters)
- Sequence formats (with and without spaces)
- Prompting strategies (simple vs verification-based)

## Requirements

- Python 3.10+
- Ollama
- Required Python packages:
  ```bash
  pip install pandas numpy tqdm matplotlib seaborn ollama
  ```

## Project Structure

- `main.py`: Primary script for running experiments
- `chat.py`: Helper module for interacting with Ollama
- `main-notebook.ipynb`: Jupyter notebook for analysis and visualization
- `results/`: Directory containing experiment results

## Usage

1. Ensure Ollama is installed and running:
```
bash
curl https://ollama.ai/install.sh | sh
```
2. Pull the required models:
```
ollama pull deepseek-r1:1.5b
ollama pull deepseek-r1:7b
ollama pull deepseek-r1:14b
ollama pull deepseek-r1:32b
```
3. Run the main script to generate results from the deepseek-r1 models:
```
python main.py
```

4. View the results in the Jupyter notebook:
```
jupyter notebook main-notebook.ipynb
```

## Results

Results are saved in the `results/` directory with:
- CSV files for raw data
- Pickle files for easy loading into pandas
- Experiment-specific subdirectories

## Analysis
The Jupyter notebook provides visualizations including:
- Accuracy by model size
- Accuracy by sequence length
- Comparison of prompting strategies
- Effect of sequence format
- Heatmaps of performance

Created for the MATS 8.0 stream.