# GenAI Deep Learning Project - Multiple Choice Question Answering

This project focuses on solving a Multiple Choice Question (MCQ) Answering task using Deep Learning and Natural Language Processing (NLP) techniques. Given a text prompt/question and a set of five possible options (A, B, C, D, E), the goal of the models developed in this project is to accurately predict the correct option.

## Directory Structure

- `dataset/`: Contains the training and testing datasets (`train.csv`, `test.csv`).
- `Notebooks/`: Contains Jupyter Notebooks used for experimentation and development.
- `Model File/`: Directory for storing trained model weights or serialized models.
- `Submission/`: Directory for generated prediction files for submission.
- `dl-23f3001984-notebook-t22026.ipynb`: The main comprehensive Jupyter Notebook containing the full workflow including EDA, model implementations, training, and evaluation.
- `venv/`: Python virtual environment containing the installed dependencies.

## Dataset

The dataset consists of multiple-choice questions in CSV format. 
- **train.csv**: Training data containing columns for `id`, `prompt`, options `A`, `B`, `C`, `D`, `E`, and the ground truth `answer`.
- **test.csv**: Test data formatted similarly for model evaluation.

## Approach & Methodology

The project is structured into three main phases as documented in the main notebook:

### 1. Exploratory Data Analysis (EDA)
Comprehensive analysis of the dataset to understand the distribution of questions and options, textual properties, length of prompts/options, and the presence of any imbalances. Techniques like TF-IDF and cosine similarity are utilized during textual analysis.

### 2. BiLSTM + Attention Implementation
A custom Deep Learning architecture is implemented using **PyTorch**. This model utilizes a Bidirectional Long Short-Term Memory (BiLSTM) network paired with an Attention mechanism to effectively process the sequence data and focus on the most relevant parts of the text when deciding the correct answer option.

### 3. Pre-Trained Transformer Model
To leverage state-of-the-art NLP performance, a pre-trained Transformer model (such as BERT from the Hugging Face `transformers` library) is implemented and fine-tuned for this specific multiple-choice question answering task.

## Dependencies

The core dependencies for this project include:
- `torch` (PyTorch)
- `transformers` (Hugging Face)
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `wandb` (Weights & Biases for experiment tracking)

To install the necessary dependencies, you can typically use:
```bash
pip install -r requirements.txt
```