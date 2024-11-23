# Fake News Detector  

Welcome to the **Fake News Detector**, a Python-based application designed to detect fake news articles using machine learning. This app provides a user-friendly interface hosted on **Streamlit**, enabling users to paste news and verify its authenticity.

---

## Features  
- **Real-Time Predictions:** Detect fake news instantly.  
- **User-Friendly Interface:** Interactive Streamlit app for easy use.  
- **Multiple Models:** Predictions based on three machine learning models:  
  - Logistic Regression (LR)  
  - Decision Tree (DT)  
  - Random Forest Classifier (RFC)

---

## Dataset  
The dataset used to train the models is publicly available on Kaggle:  
[fake-news-detection](https://www.kaggle.com/datasets/jainpooja/fake-news-detection)  

---

## Installation  

1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/your-repo-name/fake-news-detector.git
   cd Fake-News-Detector
   ```

2. **Install Dependencies:**  
   Use the following command to install the required Python packages:  
   ```bash
   pip install -r requirements.txt
   ```  

3. **Launch the App:**  
   Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```  

---

## Usage  

1. Launch the app by running `streamlit run app.py`.  
2. Copy and paste your news in the text inout area.
3. Click on the predict button. 
4. View the prediction results and confidence scores from each model.  

---

## Models  
Three machine learning models were trained and evaluated for this project:  

1. **Logistic Regression (LR):**  
   - Simple, efficient, and interpretable.  
2. **Decision Tree (DT):**  
   - Captures non-linear patterns in the data.  
3. **Random Forest Classifier (RFC):**  
   - Ensemble model providing robust and accurate predictions.  

Each model was trained on pre-processed data.

---

## File Structure  

```plaintext
fake-news-detector/
├── input/                  # Folder for dataset
├── models/                 # Trained models saved as .joblib files
├── app.py                  # Streamlit app code
├── fake_news_detection.py  # Model training code
├── requirements.txt        # Required Python packages
└── README.md               # Project documentation
```

---

## Example Screenshots  

![Home Screen](#)  
_A sample image of the app’s homepage._  

![Prediction Results](#)  
_Visualization of prediction results._  

---

## Future Enhancements  

- Add support for additional ML models (e.g., XGBoost).  
- Incorporate deep learning for enhanced accuracy.  
- Include live news scraping for real-time analysis.  
- Expand dataset support for multi-language detection.  

---

## Credits  

- **Dataset:** [Kaggle](https://www.kaggle.com/datasets/jainpooja/fake-news-detection)  
- **Libraries:** Scikit-learn, Pandas, NumPy, Joblib, re, string, Streamlit  

---  

Feel free to contribute, raise issues, or share feedback!
