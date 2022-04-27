import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#loading dataset
df = pd.read_csv("data.csv")
print('dataset information')
df.info()

#droping extra features
x = df.iloc[:, 2:-1]

#selecting target attribute
y = df.diagnosis


#Encoding target attribute in dummy variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


#selecting top corelated features to avoid overfitting
from sklearn.feature_selection import SelectKBest, f_classif
best_features = SelectKBest(score_func=f_classif)
best_features.fit(x, y)
bfdf = pd.DataFrame(data = best_features.scores_, columns=['score'])
bfdf['features'] = x.columns
bfdf = bfdf.nlargest( 30, 'score')
print()
print('correlation to target attribute')
print(bfdf)



#selecting features values
X = x[np.array(bfdf.features[0:25])].values


#split the dataset for train test purpose
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#machine learning classification model
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)


#confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print()
print('Accuracy: ', classifier.score(X_test, y_test)*100,'%')
print('Confusion matrix:')
print(cm)
print()


# cross validaiton score and standard deviation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print('K-Fold cross validaiton score')
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


