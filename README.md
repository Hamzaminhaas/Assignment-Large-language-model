# Assignment-Large-language-model
Large language model
BERT-Large Topic Classification on 20 Newsgroups
Project Description

This repository contains a topic classification system built using a fine-tuned BERT-Large model. The project investigates how modern transformer-based models perform on multi-class document classification compared to a strong traditional machine learning baseline.

The task involves assigning each document to one of 20 predefined topic categories using the well-known 20 Newsgroups dataset.

What This Project Covers

Fine-tuning a pre-trained BERT-Large model for text classification

Preparing text data using tokenisation, truncation, and padding

Training and evaluating a multi-class classifier

Implementing a TF-IDF + Logistic Regression baseline

Comparing contextual and non-contextual approaches using standard metrics

Dataset Used

The experiments are conducted on the 20 Newsgroups dataset, a widely used benchmark for topic classification tasks.

Number of classes: 20

Total documents: ~18,000

Training set: 11,314 documents

Test set: 7,532 documents

Source: Hugging Face Datasets

The dataset covers a diverse range of topics including technology, politics, science, religion, and recreation.

Model Configuration
BERT-Large Fine-Tuning

Base model: bert-large-uncased

Task type: Multi-class classification (20 labels)

Maximum token length: 512

Loss function: Cross-entropy

Optimiser: AdamW

Learning rate: 2 × 10⁻⁵

Epochs: 4

Framework: PyTorch

Training environment: Google Colab (GPU)

Traditional Baseline Model

To provide a comparison point, a classical machine learning model is also trained:

Text representation: TF-IDF

Classifier: Logistic Regression

Stop-word removal: Enabled

This baseline highlights the performance difference between frequency-based and contextual approaches.

Evaluation Metrics

Model performance is evaluated using:

Accuracy

Precision

Recall

Weighted F1-score

Confusion matrix for class-wise analysis

Weighted averaging is used to account for minor class imbalance.

Results Summary

TF-IDF + Logistic Regression

Accuracy: ~68.6%

Weighted F1-score: 0.68

Fine-tuned BERT-Large

Accuracy: ~72.1%

Weighted F1-score: 0.72

The BERT-Large model consistently outperforms the baseline, particularly for topics with overlapping vocabulary.

Key Insights

Contextual embeddings significantly improve topic understanding

Most misclassifications occur between semantically similar categories

Mild overfitting is observed due to the large capacity of BERT-Large

Limitations and Future Improvements

BERT-Large requires high computational resources

Hyperparameter tuning was limited

Possible future extensions include:

Using lighter transformer models (e.g. DistilBERT, RoBERTa)

Applying early stopping or stronger regularisation

Exploring data augmentation techniques

Purpose of This Work

This project was completed as part of an academic coursework assignment, with the objective of comparing transformer-based models against traditional machine learning approaches for document-level topic classification.
