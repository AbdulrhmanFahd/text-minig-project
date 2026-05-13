# Data Science: Dataset Integration and Sentiment Analysis

This document details the construction of the final dataset and the subsequent sentiment analysis modeling.

## 1. Dataset Construction (`build_dataset.py`)
After collecting the AI-predicted labels, they must be integrated with the original metadata and manual labels.

### Input:
- `dataset/dataset_v2.csv`: Original data with metadata (likes, nationality, manual labels).
- `dataset/labeled_results.json`: AI-generated labels from the labeling pipeline.

### Merging Logic:
1.  **AI Labels**: Extracts results from the JSON file and joins them with the CSV rows using the original index.
2.  **Manual Labels**: Identifies rows that already had manual labels (e.g., from existing datasets) and ensures they are preserved.
3.  **Deduplication**: Ensures no row is duplicated during the merge process.

### Output:
- **File**: `dataset/full_dataset.json`
- **File**: `dataset/full_dataset.csv` (for human inspection and easy loading in dataframes).

## 2. Sentiment Analysis Modeling
The final dataset is processed through an end-to-end modeling pipeline in `notebook/arabic_sa_pipeline.ipynb`.

### Step 1: Data Loading & Filtering
- **Input**: `dataset/full_dataset.json`.
- **Filtering**: The notebook extracts rows where a valid sentiment label (0 or 1) exists.
- **Handling Imbalance**: Class weights are calculated to address the natural distribution of positive vs. negative comments.

### Step 2: Exploratory Data Analysis (EDA)
- **Word Clouds**: Visualization of frequent terms in positive vs. negative classes using `arabic_reshaper` and `python-bidi`.
- **Distribution Analysis**: Breakdown of comments by nationality and like counts.

### Step 3: Text Preprocessing
- **Cleaning**: Removal of non-Arabic characters, emojis, diacritics, and extra whitespace.
- **Normalization**: Standardizing Arabic characters (e.g., alef shapes).

### Step 4: Model Architecture (AraBERT-BiLSTM)
The model combines the contextual power of BERT with the sequence-processing strengths of LSTMs:
- **Base Model**: `aubmindlab/bert-base-arabertv02` (pre-trained on large Arabic corpora).
- **Architecture**:
  - The `[CLS]` token representation from AraBERT is passed to a **Bidirectional LSTM** (BiLSTM) layer.
  - Two layers of BiLSTM with 256 hidden units.
  - Dropout layers (0.4) for regularization.
  - A final fully connected layer for binary classification.

### Step 5: Training Strategy
- **Loss Function**: **Focal Loss** is used to emphasize hard-to-classify examples and manage class imbalance.
- **Optimizer**: AdamW with a learning rate of 2e-5.
- **Scheduler**: Cosine decay with a warmup period (10% of total steps).
- **Early Stopping**: Monitored on the validation F1-score with a patience of 3 epochs.

### Step 6: Evaluation
- **Metrics**: Macro-F1 score is the primary metric, supplemented by Accuracy and AUC-ROC.
- **Output**: Generates a confusion matrix and ROC curves for the scientific paper.
- **Baseline**: Comparison against TF-IDF + Logistic Regression and SVM models.
- **Evaluation**: 
  - Metrics: Accuracy, F1-Score, Precision, and Recall.
  - Validation against a held-out test set.

## 3. Results and Model
- **Best Model**: Saved to `output/best_model.pt`.
- **Performance**: High accuracy achieved across diverse Arabic dialects found in YouTube comments.
