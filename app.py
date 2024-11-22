import streamlit as st
import joblib
import os
from fake_news_detection import output_label, wordopt, train_models
import pandas as pd

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    initial_sidebar_state="expanded",
)


# Caching for loading models and vectorizer
@st.cache_resource
def load_vectorizer():
    if os.path.exists("./models/tfidf_vectorizer.joblib"):
        return joblib.load("./models/tfidf_vectorizer.joblib")
    else:
        train_models()

@st.cache_resource
def load_model(model_name):
    file_path = f"./models/{model_name}.joblib"
    if os.path.exists(file_path):
        return joblib.load(file_path)
    else:
        "An error occurred while loading models"

# Load vectorizer and models
try:
    vectorization = load_vectorizer()
    LR = load_model("logistic_regression_model")
    DT = load_model("decision_tree_model")
    RFC = load_model("random_forest_model")
except FileNotFoundError as e:
    st.error(f"Error loading resources: {str(e)}")
    st.stop()

# Prediction function
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    predictions = [pred_LR[0], pred_DT[0], pred_RFC[0]]
    fake_count = predictions.count(0)  # Count models predicting "Fake News" (class 0)
    
    confidence = (fake_count / len(predictions)) * 100  # Confidence as a percentage
    fake_status = "Fake News" if confidence >= 50 else "Not Fake News"
    if confidence == 0.00:
        confidence = 100
    
    return (f"""
    **LR Prediction:** {output_label(pred_LR[0])}  
    **DT Prediction:** {output_label(pred_DT[0])}  
    **RFC Prediction:** {output_label(pred_RFC[0])}  
    
    **Overall Confidence:** {confidence:.2f}% sure this is {fake_status}.
    """, fake_status)

# Streamlit main function
def main():
    st.title('📰 Fake News Detector')
    st.markdown("""
    This tool analyzes news articles to determine their authenticity using advanced machine learning models.
    Enter a news article or text snippet below!
    """)
    
    news = st.text_area("Enter news article text to test authenticity:")

    if st.button('Predict'):
        if news.strip():
            try:
                with st.spinner('Processing...'):
                    result, overall_status = manual_testing(news)
                    st.markdown(result)
                    if overall_status == "Fake News":
                        st.error("🚨 The given news is likely **Fake News**.")
                    else:
                        st.success("The given news is likely **Not Fake News**.", icon="✅")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning('Please enter some text to analyze.')

if __name__ == "__main__":
    main()
