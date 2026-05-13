**AI- Based Phishing url detector**

**Overview:**
identify malicious URLs using basic feature extraction, and ml models. it starts with collecting phishing datasets –> extracting useful features from URLs and finally training models to classify them


**Goals:**

1. Learn [Phishing attacks](https://www.fortinet.com/resources/cyberglossary/types-of-phishing-attacks) basics and how mal URLs work. Download datasets from [Kaggle](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset?resource=download) or UCI. “Optionally” explore recent phishing URLs


2. Clean the dataset, remove duplicates/redundancy, and label URLs as phishing or safe. Prepare data for feature extraction.


3. Extract simple features such as URL length , no. of special characters, and HTTPS usage. “Optionally” apply text-based features like tokenization or TF-IDF.


4. Train basic machine learning models such as Logistic regression and random forest, etc. Compare their performance.


5. **“Optional”**: Build a small App (Streamlit or API) where users can input a URL and get prediction results.