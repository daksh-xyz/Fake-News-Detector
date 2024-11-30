import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from fake_news_detection import train_models
import seaborn as sns
import pandas as pd
import re
import string
import streamlit as st
import os
import numpy as np

print("Calculating evaluation metrics")

# Function for text preprocessing
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Load saved models
@st.cache_resource
def load_vectorizer():
    print("Loading vectorizer")
    if os.path.exists("./models/tfidf_vectorizer.joblib"):
        return joblib.load("./models/tfidf_vectorizer.joblib")
    else:
        return None

@st.cache_resource
def load_model(model_name):
    print(f"Loading {model_name} model")
    file_path = f"./models/{model_name}.joblib"
    if os.path.exists(file_path):
        return joblib.load(file_path)
    else:
        return None

def calculate_extra_metrics(conf_matrix):
    # Extract values from confusion matrix
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    
    # Calculate additional metrics
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    fpr = FP / (TN + FP)
    fnr = FN / (TP + FN)
    npv = TN / (TN + FN)
    fdr = FP / (FP + TP)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
    return sensitivity, specificity, fpr, fnr, npv, fdr, mcc

def plot_confusion_matrix(y_true, y_pred, model_name):
    print("Plotting graphs")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Fake", "Predicted Not Fake"], yticklabels=["Actual Fake", "Actual Not Fake"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    st.pyplot(plt)
    sensitivity, specificity, fpr, fnr, npv, fdr, mcc = calculate_extra_metrics(cm)

    metrics = {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "FPR": fpr,
        "FNR": fnr,
        "NPV": npv,
        "FDR": fdr,
        "MCC": mcc
    }
    return metrics

# Main function for evaluation
def evaluate_models():
    print("Evaluating models")
    vectorization = load_vectorizer()
    LR = load_model("logistic_regression_model")
    DT = load_model("decision_tree_model")
    RFC = load_model("random_forest_model")

    if not (vectorization and LR and DT and RFC):
        st.error("Models or vectorizer not found. Training models...")
        vectorization, LR, DT, RFC, xv_test, y_test = train_models()
    else:
        # Load dataset for evaluation
        df_fake = pd.read_csv("./input/Fake.csv")
        df_true = pd.read_csv("./input/True.csv")

        df_fake["class"] = 0
        df_true["class"] = 1
        df_merge = pd.concat([df_fake, df_true], axis=0)
        df = df_merge.drop(["title", "subject", "date"], axis=1)
        df["text"] = df["text"].apply(wordopt)

        x = df["text"]
        y = df["class"]
        _, x_test, _, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
        xv_test = vectorization.transform(x_test)

        # Generate metrics for each model
        metrics_dict = {
            "Logistic Regression": classification_report(y_test, LR.predict(xv_test), output_dict=True),
            "Decision Tree": classification_report(y_test, DT.predict(xv_test), output_dict=True),
            "Random Forest": classification_report(y_test, RFC.predict(xv_test), output_dict=True)
        }
        print("Models Evaluated!")
        models = {
            "0": LR,
            "1": DT,
            "2": RFC
        }

        model = 0

        # Create a DataFrame for each model and display it
        for model_name, metrics in metrics_dict.items():
            st.subheader(f"{model_name} Metrics")
            # Convert metrics to DataFrame
            eval_model = models.get(f"{model}")
            extra_metrics = plot_confusion_matrix(y_test, eval_model.predict(xv_test), model_name)
            df_metrics = pd.DataFrame(metrics).transpose()
            # Filter out support and format columns
            df_metrics = df_metrics[["precision", "recall", "f1-score"]].round(2)
            # Display as a table
            st.table(df_metrics)
            st.table(extra_metrics)
            model += 1

        print("Metrics displayed!")