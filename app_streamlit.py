import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score


st.title('Fraud Detection — Real vs Augmented')
st.markdown('Upload a CSV of transactions (same schema as creditcard.csv) to get predictions from saved models.')
uploaded = st.file_uploader('Upload CSV', type=['csv'])


if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write('Uploaded', df.shape)
    
    
    # Load models
    rf = joblib.load('models/fraud_model_rf.pkl')
    xgb = joblib.load('models/fraud_model_xgb.pkl')


    if 'Class' in df.columns:
        y = df['Class']
        X = df.drop('Class', axis=1)
        preds_rf = rf.predict(X)
        preds_xgb = xgb.predict(X)
        st.subheader('RandomForest report')
        st.text(classification_report(y, preds_rf))
        st.subheader('XGBoost report')
        st.text(classification_report(y, preds_xgb))
    else:
        preds = rf.predict(df)
        df['pred_rf'] = preds
        st.write(df.head())
        st.markdown('Download predictions:')
        st.download_button('Download CSV', df.to_csv(index=False), file_name='predictions.csv')