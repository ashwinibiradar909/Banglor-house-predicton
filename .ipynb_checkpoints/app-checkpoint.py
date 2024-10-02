import pandas as pd
from flask import Flask,render_template,request,redirect,url_for
import pickle
import numpy as np

app=Flask(__name__)
data=pd.read_csv('cleaned_data.csv')
pipe=pickle.load(open("RidgeModel.pkl",'rb'))

@app.route('/')
def index():
    locations=sorted(data['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=["GET","POST"])
def predict():
    pred=None
    if request.method=='POST':
        locations=request.form.get('location')
        bhk=request.form.get('bhk')
        bath=request.form.get('bath')
        sqft=request.form.get('total_sqft')
        input=pd.DataFrame([[locations,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
        prediction=pipe.predict(input)[0] * 1e5
        pred=str(np.round(prediction,2))

    return render_template('index.html',pred=pred)


if __name__=="__main__":
    app.run(debug=True)