import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
import seaborn as sns
from matplotlib import pyplot as plt

mlflow.set_tracking_uri('http://127.0.0.0.1:5000')

x,y=make_classification(n_classes=2,n_informative=2,n_redundant=2,n_samples=200,random_state=5)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)

n=20
m=10

mlflow.set_experiment('mlflow-dagshub1')

with mlflow.start_run():

    r=RandomForestClassifier(n_estimators=n,max_depth=m)
    r.fit(x_train,y_train)
    y_pred=r.predict(x_test)
    acc=accuracy_score(y_test,y_pred)

    c=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(c,cmap='Blues',annot=True)
    plt.xlabel('actual')
    plt.ylabel('predict')
    plt.title('confusion matrix')
    plt.savefig('m.png')

    mlflow.log_metric('accuracy',acc)
    mlflow.log_param('n_estimator',n)
    mlflow.log_param('max_depth',m)
    mlflow.log_artifact('m.png')
    mlflow.set_tag('authore','ayush')
    mlflow.set_tag('model','Randomforest')
    mlflow.log_artifact(__file__)
    mlflow.sklearn.Model(r,'Randomforest')