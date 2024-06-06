import pickle
import pandas as pd


# Fetch Data
# raw_data = {
#     "PassengerId" : [892],
#     "Pclass" : [3],
#     "Name" : ["Kelly, Mr. James"],
#     "Sex" : ["male"],
#     "Age" : [34.5],
#     "SibSp" : [0],
#     "Parch" : [0],
#     "Ticket" : [330911],
#     "Fare" : [7.8292],
#     "Cabin" : [None],
#     "Embarked" : ["Q"]
# }



# Data Cleansing
def clean(data):
    data = data.drop(["Ticket", "Cabin", "Name", "PassengerId"], axis=1)
    cols = ["SibSp", "Parch" ,"Age", "Fare"]
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)
    data["Pclass"].fillna(3, inplace=True)
    data["Embarked"].fillna("U", inplace=True)
    return data

def fetchAndCleanData(raw_data):
    data = pd.DataFrame.from_dict(raw_data)
    clean_data = clean(data)
    return clean_data

# Data Conversion
def transformer(col, element):
    transformer_data = {
        'Sex': {
            'male':0, 
            'female':1
        },
        'Embarked' : {
            'c': 0,
            'q': 1,
            's': 2,
            'u': 3
        }
    }
    
    return transformer_data[col][element.lower()]

def transformData(data):
    cols = ["Sex", "Embarked"]
    for col in cols:
        data[col] = data[col].apply(lambda x : transformer(col,x))
    return data

# Prediction
def prediction(raw_data):
    clean_data = fetchAndCleanData(raw_data)
    data = transformData(clean_data)
    model = pickle.load(open('model.pkl','rb'))
    return model.predict(data)