# Spam Message Prediction System

## Run locally
1. Install Python 3.10+.
2. In this folder run `python -m venv .venv`.
3. Activate it (`.venv\\Scripts\\activate` on Windows).
4. Run `pip install -r requirements.txt`.
5. Run `streamlit run app.py`.

## What is included
- `app.py`: interactive Streamlit application.
- `data/sample_messages.csv`: small illustrative dataset so the app works immediately.
- TF-IDF text vectorization + Multinomial Naive Bayes classifier.
- Hold-out evaluation display and message prediction.

## Important academic limitation
The included 24-row sample is for demonstrating the complete pipeline, not for claiming high real-world accuracy. For a proper experiment, download the full **UCI SMS Spam Collection** (5,574 labelled messages) from https://archive.ics.uci.edu/dataset/228/sms+spam+collection. Convert its tab-separated file to `data/sample_messages.csv` with columns `label,message` (labels `ham` and `spam`), then rerun the app. Record the new test accuracy and confusion matrix; do not copy an accuracy figure from another project.

## Method
1. Split labelled messages into training and test partitions.
2. TF-IDF converts text into weighted token features.
3. Multinomial Naive Bayes learns class probabilities.
4. Evaluate predictions on held-out messages.
5. Use the trained pipeline to classify user input.

## Safety and scope
The classifier is an educational prototype. False positives and false negatives are possible. It does not inspect email headers, attachments, URLs, or sender reputation.

## Dataset citation
Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection*. UCI Machine Learning Repository. DOI: 10.24432/C5CC84. Dataset license: CC BY 4.0. Cite the dataset in your report if you use it.
