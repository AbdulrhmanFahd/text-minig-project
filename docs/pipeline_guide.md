# End-to-End Arabic Sentiment Analysis Pipeline

This document provides a comprehensive guide to the entire data preparation and modeling workflow.

## Project Structure
```text
.
├── dataset/             # Raw and processed datasets
│   ├── dataset_v2.csv   # Source data
│   ├── full_dataset.json # Final labeled dataset
│   └── ...
├── docs/                # Project documentation
│   ├── text_mining.md   # Labeling pipeline details
│   ├── data_science.md  # Dataset building & modeling details
│   └── pipeline_guide.md # This file
├── notebook/            # Analysis notebooks
│   └── arabic_sa_pipeline.ipynb
├── script/             # Automation scripts
│   ├── batch_labeler.py
│   └── build_dataset.py
└── output/              # Model artifacts
    └── best_model.pt
```

## Workflow Summary

### Phase 1: Data Acquisition & Consolidation
Raw data was consolidated from **150 separate Excel files** into a unified format in `dataset/dataset_v2.csv`.

### Phase 2: Automated Labeling (Text Mining)
Using the **Gemini Batch API**, we labeled ~130k comments. This process is documented in [text_mining.md](./text_mining.md).
- **Tool**: `script/batch_labeler.py`
- **Output**: `dataset/labeled_results.json`

### Phase 3: Dataset Integration
The predicted labels were merged with existing manual labels and metadata to create the final training set. This is documented in [data_science.md](./data_science.md).
- **Tool**: `script/build_dataset.py`
- **Output**: `dataset/full_dataset.json`

### Phase 4: Model Training
A fine-tuned AraBERT-BiLSTM model was trained on the final dataset.
- **Tool**: `notebook/arabic_sa_pipeline.ipynb`
- **Output**: `output/best_model.pt`

## Getting Started
1. Ensure your `.env` file contains a valid `GEMINI_API_KEY`.
2. Run labeling jobs: `python script/batch_labeler.py --submit`.
3. Collect results: `python script/batch_labeler.py --collect`.
4. Build dataset: `python script/build_dataset.py`.
5. Open the notebook in the `notebook/` folder to train or evaluate the model.
