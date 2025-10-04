import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from utils import load_data, preprocess_for_model, split_data




def train_and_eval(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    proba = rf.predict_proba(X_test)[:,1]
    print('RandomForest report:')
    print(classification_report(y_test, preds))
    print('AUC:', roc_auc_score(y_test, proba))
    return rf




def main():
    df = load_data()
    X, y = preprocess_for_model(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    
    print('Training on real data...')
    rf_baseline = train_and_eval(X_train, y_train, X_test, y_test)
    joblib.dump(rf_baseline, 'models/fraud_model_rf_baseline.pkl')
    
    
    # Training on augmented data
    aug = pd.read_csv('outputs/augmented_dataset.csv')
    X_aug, y_aug = preprocess_for_model(aug)
    X_train_a, X_test_a, y_train_a, y_test_a = split_data(X_aug, y_aug)
    
    
    print('Training on augmented data...')
    rf_aug = train_and_eval(X_train_a, y_train_a, X_test_a, y_test_a)
    joblib.dump(rf_aug, 'models/fraud_model_rf.pkl')
    
    
    # XGBoost example
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train_a, y_train_a)
    joblib.dump(xgb, 'models/fraud_model_xgb.pkl')
    print('Saved models to models/ folder')


if __name__ == '__main__':
    main()