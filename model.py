from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, recall_score
 
def Logistic(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f1_score(y_test,y_pred,average='binary'))
    print(recall_score(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    return model.coef_