# Fake News Detection Using RNN

A Natural Language Processing (NLP) project that uses a Recurrent Neural Network (RNN) to classify news text as **genuine or fake**. The project covers text preprocessing, tokenization, sequence padding, model training, and prediction on unseen data.

## Project Overview

Fake news detection is a text classification task that aims to identify potentially misleading or fabricated news content. In this project, textual data is cleaned and transformed into numerical sequences before being passed to a deep learning model for classification.

The model is trained on labeled news data and then used to generate predictions for a separate test dataset.

## Objectives

- Build a text preprocessing pipeline for news data.
- Convert text into numerical sequences suitable for deep learning.
- Train an RNN-based text classification model.
- Evaluate model performance using accuracy and loss.
- Generate predictions and export them to a CSV submission file for Kaggle.

## Technologies Used

- **Python** — Core programming language
- **Pandas & NumPy** — Data handling and numerical operations
- **NLTK** — Text preprocessing and linguistic processing
- **TensorFlow / Keras** — Deep learning model development
- **Scikit-learn** — Dataset splitting and label encoding

## Workflow

### 1. Data Loading

The training and test datasets are loaded from CSV files using Pandas.

- `xy_train.csv` — Training data containing text and labels
- `x_test.csv` — Test data containing text for prediction

The training dataset includes a text column and a label column.

### 2. Text Preprocessing

A custom text-cleaning function prepares the text for modeling through:

- Converting text to lowercase
- Removing extra whitespace and non-word characters
- Tokenizing text into individual words
- Applying lemmatization using WordNet
- Removing English stopwords
- Filtering out words shorter than three characters

The cleaned text is stored in a new column for further processing.

### 3. Dataset Preparation

The text and labels are separated into features (`X`) and targets (`y`).

The labels are encoded using Scikit-learn's `LabelEncoder`, and the labeled data is split into training and validation subsets.

### 4. Text Representation

Since neural networks require numerical inputs, the cleaned text is converted into sequences of integers using the Keras `Tokenizer`.

- Vocabulary limit: 20,000 words
- Maximum sequence length: 40 tokens
- Padding: Applied to ensure all sequences have the same length

### 5. RNN Model Architecture

The model is built using the Keras Functional API.

| Layer | Purpose |
|---|---|
| Input | Accepts padded text sequences |
| Embedding | Learns vector representations of words |
| SimpleRNN | Captures sequential patterns in the text |
| Dropout | Helps reduce overfitting |
| Dense (ReLU) | Learns higher-level representations |
| Dropout | Adds regularization |
| Dense (Softmax) | Produces class probabilities |

### 6. Model Training

The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss.

| Hyperparameter | Value |
|---|---:|
| Embedding dimension | 100 |
| RNN units | 16 |
| Batch size | 32 |
| Epochs | 6 |
| Learning rate | 0.0001 |
| Maximum sequence length | 40 |

Training includes validation to monitor performance across epochs.

### 7. Prediction and Submission

After training, the model processes the separate test dataset using the same text preprocessing, tokenization, and padding steps.

The predicted class is selected using the highest output probability, and the results are exported to `submission.csv`.

## Project Structure

```text
fake-news-detection/
├── xy_train.csv
├── x_test.csv
├── notebook.ipynb
├── submission.csv
└── README.md
```


## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd fake-news-detection
```

### 2. Install dependencies

```bash
pip install numpy pandas nltk scikit-learn tensorflow
```

### 3. Download the required NLTK resources

Run the following in Python:

```python
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 4. Prepare the datasets

Place `xy_train.csv` and `x_test.csv` in the expected working directory and ensure they contain the required `text` column. The training dataset must also contain a `label` column.

### 5. Run the notebook

Open `notebook.ipynb` in Jupyter Notebook or Google Colab and execute the cells in order.

## Evaluation

The model is evaluated using:

- **Accuracy** — The proportion of correctly classified examples.
- **Loss** — The model's classification error according to the selected loss function.

Actual evaluation results should be added here after verifying the model's performance on an appropriate held-out dataset.

## Limitations and Future Improvements

Potential improvements include:

- Comparing RNN performance with LSTM and GRU architectures.
- Experimenting with different sequence lengths and vocabulary sizes.
- Adding precision, recall, F1-score, and a confusion matrix.
- Improving preprocessing and handling missing or duplicate records.
- Using pretrained word embeddings or transformer-based models.
- Evaluating on an independent test set to obtain a more reliable estimate of generalization.

## Disclaimer

This project is an educational text-classification experiment. Its predictions should not be treated as definitive proof that a news article is true or false. Model performance depends on the quality, representativeness, and labeling of the training data.

## Author

**Malak Essam Kamal**

Electronics & Communication Engineering Graduate | AI & Machine Learning

