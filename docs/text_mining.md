# Text Mining: Automated Data Labeling Pipeline

This document describes the automated process used to label the Arabic YouTube comment dataset using Large Language Models (LLMs).

## Overview
The goal was to label approximately 130,000 unlabeled Arabic comments with sentiment labels (Positive/Negative) and confidence scores. This was achieved using the **Gemini Batch API** for cost-efficiency and high throughput.

## 1. Data Acquisition and Consolidation
The raw data for this study consisted of YouTube comments distributed across **150 separate Excel files**. 

### Consolidation Process:
1.  **Extraction**: All files were extracted from the source archive (`dataset/150FilesOfYouTubeComments.rar`).
2.  **Merging**: A Python script was used to iterate through all Excel files and consolidate them into a single master CSV file.
3.  **Deduplication**: Initial cleaning was performed to remove exact duplicate comments across different video sources.
4.  **Formatting**: The resulting file, `dataset/dataset_v2.csv`, serves as the primary input for the labeling pipeline.

## 2. Automated Data Labeling (`batch_labeler.py`)
The pipeline uses a custom script to interface with the Google AI Studio Batch API to label approximately 130,000 comments.

### Key Features:
- **Few-Shot Prompting**: The model is provided with 4 representative examples (2 positive, 2 negative) to establish the labeling standard.
- **Batching**: Comments are grouped into requests (5 comments per request) and then into batch jobs (up to 2,000 requests per job).
- **Dual-Key Support**: Allows running parallel labeling jobs using multiple API keys to overcome quota limits.
- **Error Recovery**: Implements multiple JSON parsing strategies to recover labels even from truncated or slightly malformed model responses.
- **Confidence Filtering**: Every label comes with a confidence score (0.0-1.0). Labels with confidence < 0.75 are flagged for review.

### Usage:
```bash
# Submit new batch jobs for labeling
python script/batch_labeler.py --submit

# Check the status of active jobs
python script/batch_labeler.py --status

# Collect results once jobs are completed
python script/batch_labeler.py --collect
```

## 3. Output Data
- **File**: `dataset/labeled_results.json`
- **Content**: A JSON array of objects containing:
  - `original_index`: The row index in the source CSV.
  - `label`: Predicted sentiment (0 = Negative, 1 = Positive).
  - `confidence`: Model confidence score.
  - `low_confidence`: Boolean flag if confidence is below 0.75.

## 4. Scientific Significance
This automated labeling approach allowed for the creation of a massive, high-quality Arabic sentiment dataset in a fraction of the time required for manual labeling, while maintaining consistency through structured prompting.
