# Enhancing Fraud Detection using Synthetic Transactions (CTGAN)
It uses CTGAN (from the sdv library) to generate synthetic fraud transactions, augments the original dataset, trains ML models (RandomForest &amp; XGBoost), and evaluates performance improvement. It also contains a Streamlit demo to explore results and compare models.

1) Go to the src and get the dataset of creditcard

2) Then run "pip install -r requirements.txt" to get all the libraries.

3) Run scripts in order:

    python 1_EDA.py
    python 2_CTGAN_Training.py
    python 3_Model_Training.py
    python 4_Evaluation.py

4) Run the demo (optional):
    streamlit run app_streamlit.py

You can get the overview from:
    Final_Report.pdf
    Fraud_Detection_Slides.pptx
