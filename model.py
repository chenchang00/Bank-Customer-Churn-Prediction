from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, recall_score, classification_report, precision_recall_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import xgboost as xgb
 
def Logistic(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"auc:{roc_auc_score(y_test, y_pred_proba)}") 
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

    f1_scores = 2 * (precision * recall) / (precision + recall)

    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]
    print(f"best f1:{np.max(f1_scores)}")
    print(f"best_threshold:{best_threshold}")
    

    return model.coef_

def rf(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    feature_importances = rf_model.feature_importances_
    print(f1_score(y_test,y_pred,average='binary'))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    param_dist = {
        'n_estimators': randint(100, 500),
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5),
        'bootstrap': [True, False]
    }

    # Perform Randomized Search
    random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, 
                                    n_iter=100, cv=5, n_jobs=-1, scoring='f1', verbose=2, random_state=42)
    random_search.fit(X_train, y_train)

    best_random_model = random_search.best_estimator_
    y_pred_random = best_random_model.predict(X_test)
    print(f1_score(y_test,y_pred_random,average='binary'))
    print(confusion_matrix(y_test, y_pred_random))
    print(classification_report(y_test, y_pred_random))

    return feature_importances

def interpret_rf(feature_importances, features):
    feature_importances_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances_df)
    plt.title('Feature Importances in Random Forest')
    plt.show()

def xgboost(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f1)
