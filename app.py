# import flask
# import numpy as np
from flask import Flask, request, render_template
import pickle
from predict import prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    raw_data = {
    "PassengerId" : [request.form["PassengerId"]],
    "Pclass" : [request.form["Pclass"]],
    "Name" : [request.form["Name"]],
    "Sex" : [request.form["Sex"]],
    "Age" : [int(request.form["Age"])],
    "SibSp" : [int(request.form["SibSp"])],
    "Parch" : [int(request.form["Parch"])],
    "Ticket" : [request.form["Ticket"]],
    "Fare" : [float(request.form["Fare"])],
    "Cabin" : [request.form["Cabin"]],
    "Embarked" : [request.form["Embarked"]]
}
    prediction_result = prediction(raw_data)

    # output = round(prediction[0], 2)
    if prediction_result:
        return render_template('index.html', prediction_text='Yaay !! You Survived')
    else:
        return render_template('index.html', prediction_text='Sorry !! You wouldnt have Survived')
    
# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port='5000')