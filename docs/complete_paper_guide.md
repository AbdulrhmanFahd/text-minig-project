# Methodology: Arabic Sentiment Analysis Pipeline

This document provides a consolidated technical description of the methodologies used in this study, from data acquisition to model evaluation. It is designed to be integrated into the methodology section of a scientific paper.

---

## 1. Data Acquisition and Preprocessing
The research began with the collection of raw user comments from various Arabic-language YouTube channels.

### 1.1 Data Consolidation
- **Initial Data**: 150 separate Excel files containing raw comments and metadata.
- **Consolidation**: A Python-based automation script merged these files into a single master dataset (`dataset_v2.csv`).
- **Deduplication**: Exact duplicates and empty comments were removed to ensure data quality.

## 2. Automated Labeling (Text Mining)
Given the volume of data (~130,000 comments), a manual labeling approach was infeasible. We developed an automated pipeline utilizing Large Language Models (LLMs).

### 2.1 Gemini Batch API Integration
- **Model**: Gemini 1.5 Flash.
- **Prompting Strategy**: Few-shot prompting was employed with representative examples of Positive and Negative sentiments to guide the model.
- **Batch Processing**: Comments were processed in groups of 2,000 per batch job to optimize throughput and cost.
- **Quality Control**: Every prediction included a confidence score. Labels with a confidence score below 0.75 were excluded or flagged for manual verification.

## 3. Dataset Construction
The AI-generated labels were integrated with existing ground-truth data to build the final research corpus.

- **Merging Logic**: The `build_dataset.py` script aligned batch results with original CSV metadata using row indices.
- **Final Format**: Data was exported to `full_dataset.json` (for model training) and `full_dataset.csv` (for analysis).

## 4. Sentiment Analysis Modeling
The core of the study involves a hybrid deep learning architecture designed for Arabic text.

### 4.1 Text Preprocessing
- **Cleaning**: Removal of non-Arabic characters, emojis, and diacritics.
- **Normalization**: Standardizing character forms (e.g., alef-hamza variants).
- **Tokenization**: Utilizing the AraBERT specific tokenizer which handles Arabic morphological complexity.

### 4.2 Model Architecture (AraBERT-BiLSTM)
We implemented a multi-stage architecture:
1. **AraBERT Layer**: We used `aubmindlab/bert-base-arabertv02` to extract contextualized embeddings.
2. **BiLSTM Layer**: A two-layer Bidirectional LSTM (512 units) processes the BERT output to capture long-range temporal dependencies in the text.
3. **Classification Head**: Dropout layers (0.4) and a fully connected layer with a ReLU activation for binary sentiment classification.

### 4.3 Training and Evaluation
- **Loss Function**: Focal Loss was implemented to address class imbalance and focus the model on difficult examples.
- **Optimization**: AdamW optimizer with a cosine learning rate scheduler.
- **Evaluation Metrics**: The model was evaluated on a held-out test set using Accuracy, Macro-F1, and AUC-ROC scores.
- **Baselines**: The performance was compared against traditional Machine Learning baselines, including TF-IDF + Logistic Regression and SVM.

---

## Conclusion
This pipeline establishes a robust framework for large-scale Arabic sentiment analysis, combining efficient LLM-based labeling with state-of-the-art deep learning architectures.
