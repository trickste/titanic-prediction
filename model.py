import pandas as pd
import os
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import sys


#Data Injestion
data_file_path = os.path.join(os.getcwd(), "train.csv")
test_file_path = os.path.join(os.getcwd(), "test.csv")
data = pd.read_csv(data_file_path)
test = pd.read_csv(test_file_path)
test_ids = test["PassengerId"]

#Data Cleaning and Nan management
def clean(data):
    data = data.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis=1)
    cols = ["SibSp", "Parch" ,"Age", "Fare"]
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)
    data["Embarked"].fillna("U", inplace=True)
    return data


data=clean(data)
test=clean(test)

#Data conversion to INT 
le = preprocessing.LabelEncoder()
cols=["Sex", "Embarked"]
for col in cols:
    data[col] = le.fit_transform(data[col])
    test[col] = le.transform(test[col])


#Data Split
y = data.loc[:,"Survived"]
X = data.drop(["Survived"], axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


#Train Model
def trainModel(X_train, y_train):
    clr = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
    pickle.dump(clr, open('model.pkl','wb'))
    print("Model has been created")


#Test Model
def testModel(X_val, y_val):
    try:
        model = pickle.load(open('model.pkl','rb'))
        test_prediction = model.predict(X_val)
        a_score = accuracy_score(y_val, test_prediction)*100
        if a_score> 70:
            print("Test is Successful, accuracy was high: {:.2f}%".format(a_score) )
        else:
            print("Accuracy is low: {:.2f}%".format(a_score))
    except:
        print("Test Fail, model not found")
        exit(1)


if sys.argv[1] == "test":
    testModel(X_val, y_val)
else:
    trainModel(X_train, y_train)