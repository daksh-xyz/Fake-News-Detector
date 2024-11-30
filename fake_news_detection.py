import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import string
import os

print("running dependency")

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


def train_models():
    print("Training models")
    # Load and preprocess the dataset
    df_fake = pd.read_csv("./input/Fake.csv")
    df_true = pd.read_csv("./input/True.csv")

    df_fake["class"] = 0
    df_true["class"] = 1

    df_merge = pd.concat([df_fake, df_true], axis=0)
    df = df_merge.drop(["title", "subject", "date"], axis=1)
    df["text"] = df["text"].apply(wordopt)

    # Define features and target
    x = df["text"]
    y = df["class"]

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    # Vectorize the text
    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)

    # Train models
    LR = LogisticRegression()
    LR.fit(xv_train, y_train)

    DT = DecisionTreeClassifier()
    DT.fit(xv_train, y_train)

    RFC = RandomForestClassifier(random_state=0)
    RFC.fit(xv_train, y_train)

    # Define the folder path
    folder_path = "models"

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    vector_file = "tfidf_vectorizer.joblib"
    LR_file = "logistic_regression_model.joblib"
    DT_file = "decision_tree_model.joblib"
    RFC_file = "random_forest_model.joblib"

    vector_path = os.path.join(folder_path, vector_file)
    LR_path = os.path.join(folder_path, LR_file)
    DT_path = os.path.join(folder_path, DT_file)
    RFC_path = os.path.join(folder_path, RFC_file)


    # Save models and vectorizer
    joblib.dump(vectorization, vector_path)
    joblib.dump(LR, LR_path)
    joblib.dump(DT, DT_path)
    joblib.dump(RFC, RFC_path)

    print("Models and vectorizer saved successfully!")


def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"