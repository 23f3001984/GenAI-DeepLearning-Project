# GenAI MCQ Solver - Deep Learning Question Answering

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lstmattention-bbee7rdgexgjl2ihdvkohp.streamlit.app/)

This project focuses on solving a Multiple Choice Question (MCQ) Answering task using Deep Learning and Natural Language Processing (NLP) techniques. Given a text prompt/question and a set of five possible options (A, B, C, D, E), the models developed in this project accurately predict the correct option.

## 🚀 Live Demo

A live, interactive web application has been deployed using Streamlit, allowing users to input their own MCQs and get real-time predictions from the model.

**[👉 Click here to try the live application!](YOUR_STREAMLIT_LINK_HERE)** <!-- REPLACE 'YOUR_STREAMLIT_LINK_HERE' WITH YOUR ACTUAL DEPLOYMENT LINK -->

## 📁 Directory Structure

- `dataset/`: Contains the training and testing datasets (`train.csv`, `test.csv`).
- `Notebooks/`: Contains Jupyter Notebooks used for experimentation and development.
- `Model File/`: Directory for storing trained model weights or serialized models.
- `Submission/`: Directory for generated prediction files for submission.
- `dl-23f3001984-notebook-t22026.ipynb`: The **main comprehensive Jupyter Notebook** containing the core workflow, including EDA, custom BiLSTM + Attention implementation, Transformer fine-tuning, and evaluation.
- `notebook95b58f4134-rag-implementation.ipynb`: Supplementary notebook containing the implementation of the FAISS-based Retrieval-Augmented Generation (RAG) pipeline.
- `rag_artifacts/`: Directory containing generated FAISS indices and chunked data for the RAG pipeline.
- `venv/`: Python virtual environment containing the installed dependencies.

## 🧠 Approach & Methodology

The project is structured into several phases, with a primary focus on deep learning architectures:

### 1. Exploratory Data Analysis (EDA)
Comprehensive analysis of the dataset to understand the distribution of questions and options, textual properties, length of prompts/options, and the presence of any imbalances. Techniques like **TF-IDF and cosine similarity** are utilized during textual analysis to inform model design.

### 2. Custom Deep Learning Architecture (BiLSTM + Attention)
A custom Deep Learning architecture is implemented using **PyTorch**. This model utilizes a Bidirectional Long Short-Term Memory (**BiLSTM**) network paired with an **Attention mechanism** to effectively process the sequence data and contextually focus on the most relevant parts of the text when deciding the correct answer.

### 3. Pre-Trained Transformer Model Fine-Tuning
To leverage state-of-the-art NLP performance, pre-trained Transformer models (such as **BERT** from the Hugging Face `transformers` library) are implemented and fine-tuned specifically for this multiple-choice question answering downstream task.

### 4. Experiment Tracking
**Weights & Biases (W&B)** is integrated throughout the training lifecycle for systematic experiment tracking, hyperparameter optimization, and logging across multiple training iterations.

### 5. Context Enrichment (Supplementary RAG)
As a minor supplementary feature to augment prompt accuracy, a lightweight **Retrieval-Augmented Generation (RAG)** pipeline is implemented. It leverages **FAISS** vector indexing and Hugging Face sentence embeddings to retrieve relevant context from Wikipedia data dumps.

## 🛠️ Technologies Used

- **Languages:** Python
- **Deep Learning Frameworks:** PyTorch
- **NLP & Transformers:** Hugging Face, BERT, TF-IDF
- **Vector Search:** FAISS
- **Experiment Tracking:** Weights & Biases (W&B)
- **Deployment:** Streamlit
- **Data Science Stack:** Pandas, NumPy, Scikit-learn, Matplotlib

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. **Activate your virtual environment (if using one):**
   ```bash
   source venv/bin/activate  # On macOS/Linux
   # or
   .\venv\Scripts\activate   # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
