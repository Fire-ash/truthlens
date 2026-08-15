# TruthLens

AI-Based Fake News Detection and Classification System

TruthLens is a machine-learning application that analyzes news text and classifies it as **REAL**, **FAKE**, or **UNCERTAIN**. The system uses Natural Language Processing with TF-IDF feature extraction and Logistic Regression, with a Streamlit interface for interactive analysis.

## Project Overview

Fake news detection can be approached as a text-classification problem. TruthLens learns linguistic patterns from labeled real and fake news examples and estimates which class a new piece of text most closely resembles.

The application provides a prediction and confidence score through a web-based interface.

> **Important:** TruthLens is a classification system, not an independent fact-checking engine. It does not verify claims against external sources.

## Features

* News text analysis through a Streamlit web interface
* REAL / FAKE / UNCERTAIN classification
* TF-IDF based text feature extraction
* Logistic Regression classification
* Probability-based prediction
* Configurable decision threshold
* Configurable uncertainty band
* Word-count feedback for short inputs
* Model performance display
* Dataset and model limitations documented
* Command-line prediction support
* Automated evaluation and testing

## System Architecture

```text
News Text
    |
    v
Text Preprocessing
    |
    v
TF-IDF Vectorization
    |
    v
Logistic Regression
    |
    v
Fake Probability
    |
    v
Decision Threshold
    |
    +------------------+
    |        |         |
    v        v         v
  REAL   UNCERTAIN    FAKE
    |
    v
Streamlit Interface
```

## Machine Learning Pipeline

### 1. Text Preprocessing

Input text is normalized before feature extraction. The preprocessing pipeline handles:

* Conversion to lowercase
* URL removal
* Email removal
* Non-ASCII artifact removal
* Whitespace normalization
* Optional removal of source-specific artifacts

The same preprocessing pipeline is used during training and prediction.

### 2. TF-IDF

TF-IDF converts text into numerical feature vectors.

The implementation uses word n-grams so that both individual words and short word combinations can contribute to classification.

### 3. Logistic Regression

Logistic Regression receives the TF-IDF representation and estimates the probability that the input belongs to the FAKE class.

The output probability is then passed through the project's decision logic.

### 4. Uncertainty Handling

Instead of forcing every prediction into REAL or FAKE, TruthLens supports an UNCERTAIN class.

With the default configuration:

```text
FAKE threshold: 0.50
Uncertainty band: 0.10

REAL:       probability < 0.45
UNCERTAIN:  0.45 to 0.55
FAKE:       probability > 0.55
```

This allows the system to abstain when the prediction is close to the decision boundary.

## Dataset

The project uses labeled real and fake news data stored in:

```text
data/
├── True.csv
└── Fake.csv
```

The dataset contains:

```text
Original samples:       1,998
After deduplication:    1,991

REAL:                     992
FAKE:                     999
```

The data is divided using a stratified train/test split.

## Model Evaluation

The trained model was evaluated using:

* Accuracy
* Macro F1
* Precision
* Recall
* ROC-AUC
* Average Precision
* Confusion Matrix
* Cross-validation

The current leakage-controlled run produced approximately:

```text
Holdout Accuracy:  99.5%
Macro F1:          99.5%
ROC-AUC:           1.000
```

### Important Evaluation Limitation

The dataset contains significant source-label confounding.

The available `subject` groups are distributed almost perfectly according to the target label:

```text
politicsNews -> REAL
News         -> FAKE
```

This means that high evaluation scores can partly reflect source-specific patterns rather than genuine misinformation detection ability.

The project therefore treats the reported accuracy as a dataset-specific benchmark rather than a real-world accuracy claim.

Reuters-specific source markers were also removed during the current training run to reduce one obvious source shortcut.

## Application

The Streamlit application provides:

* News text input
* Word count
* Prediction result
* Confidence score
* Advanced decision controls
* Model performance information
* Explanation of the classification approach
* Responsible-use limitations

Run the application with:

```bash
streamlit run src/streamlit_app.py
```

Then open the local Streamlit address shown in the terminal.

## Command-Line Prediction

A prediction can also be generated from the command line:

```bash
python src/detect_fake_news.py \
  --text "Officials announced a new economic policy today."
```

JSON output can be requested with:

```bash
python src/detect_fake_news.py \
  --text "Officials announced a new economic policy today." \
  --json
```

## Project Structure

```text
TruthLens/
├── data/
│   ├── True.csv
│   └── Fake.csv
│
├── docs/
│   ├── data_statement.md
│   ├── model_card.md
│   └── security.md
│
├── outputs/
│   ├── charts/
│   ├── metrics.json
│   ├── leakage_report.json
│   ├── source_confounding_report.json
│   └── pipeline.joblib
│
├── src/
│   ├── detect_fake_news.py
│   ├── evaluation.py
│   ├── model_compat.py
│   ├── streamlit_app.py
│   ├── text_clean.py
│   └── train_model.py
│
├── tests/
├── README.md
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train_model.py --strip-source-artifacts
```

Launch the application:

```bash
streamlit run src/streamlit_app.py
```

## Testing

Run the test suite with:

```bash
pytest
```

Python source compilation can be checked with:

```bash
python -m compileall src tests
```

## Limitations

TruthLens has several important limitations:

* The dataset is relatively small.
* The dataset contains source-label confounding.
* The model does not independently verify factual claims.
* Performance on unseen publishers may differ substantially.
* Very short inputs provide fewer useful features.
* High benchmark accuracy does not imply real-world fake-news detection accuracy.
* The system should not be used for high-stakes decisions.

## Future Improvements

Potential improvements include:

* Training on a larger and source-balanced dataset
* Testing on publishers completely absent from training
* Adding external evidence retrieval
* Adding explainable feature analysis
* Improving probability calibration
* Evaluating transformer-based NLP models
* Adding out-of-distribution detection
* Deploying the application as a public web service

## Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit
* Pytest

## Author

**Udit Bisht**

## License

This project is intended for educational and academic purposes.
