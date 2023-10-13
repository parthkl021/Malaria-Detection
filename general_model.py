from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def LogisticModel(x_train,x_test,y_train,y_test):
    model = LogisticRegression(max_iter = 500)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = model.score(x_test, y_test)
    return score
    

def SVMModel(x_train,x_test,y_train,y_test):
    clf_svc = SVC().fit(x_train, y_train)
    y_pred = clf_svc.predict(x_test)
    return accuracy_score(y_test, y_pred)


def StandardScaling(X,Y):
    standard = StandardScaler()
    standard = standard.fit(X)
    return standard.transform(Y)

    
    
