import numpy as np
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


#Predicting the Test set results
# 1 is for Malignant and 0 for Benign
ts = pd.DataFrame(data=y_test, columns=['exact result'])
ts['predicted result'] = classifier.predict(X_test)
print()
print('Predicting the Test set results')
print(ts)
print()


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


#Predicting a new result
#features
['concave points_worst', 'perimeter_worst', 'concave points_mean', 'radius_worst', 'perimeter_mean', 'area_worst', 
'radius_mean', 'area_mean', 'concavity_mean', 'concavity_worst', 'compactness_mean', 'compactness_worst', 'radius_se',
'perimeter_se', 'area_se', 'texture_worst', 'smoothness_worst', 'symmetry_worst', 'texture_mean', 'concave points_se',
'smoothness_mean', 'symmetry_mean', 'fractal_dimension_worst', 'compactness_se', 'concavity_se']


input = [2.430e-01, 1.525e+02, 1.279e-01, 2.357e+01, 1.300e+02, 1.709e+03,
        1.969e+01, 1.203e+03, 1.974e-01, 4.504e-01, 1.599e-01, 4.245e-01,
        7.456e-01, 4.585e+00, 9.403e+01, 2.553e+01, 1.444e-01, 3.613e-01,
        2.125e+01, 2.058e-02, 1.096e-01, 2.069e-01, 8.758e-02, 4.006e-02,
        3.832e-02]

print()
print('New result: ',classifier.predict([input]))
#Model predicted right. 1 is for Malignant (M), 0 is for Benign (B)
