import pandas as pd
import numpy as np
import sklearn
import joblib
from flask import Flask,render_template,request
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from gevent.pywsgi import WSGIServer
import os
#import flasgger
#from flasgger import Swagger

app=Flask(__name__)
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
pickel=open(os.path.join(THIS_FOLDER, 'diabetes_pickle.pkl'),"rb")

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict/individual',methods=["Get"])
def individual():
    return render_template("individual.html")

@app.route('/results',methods=['GET','POST'])
def results():
    model_prediction=69
    if request.method =='POST':
        try:
            glucose=float(request.form['glucose'])
            pregnancies=float(request.form['pregnancies'])
            bloodPressure=float(request.form['bloodPressure'])
            skinThickness=float(request.form['skinThickness'])
            insulin=float(request.form['insulin'])
            bmi=float(request.form['bmi'])
            dpf=float(request.form['dpf'])
            age=float(request.form['age'])
            pred_args=[pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,dpf,age]
            pred_arr=np.array(pred_args)
            print(pred_arr)
            preds=pred_arr.reshape(1,-1)
            
            model=joblib.load(pickel)
            model_prediction=model.predict(preds)			
            model_prediction=round(float(model_prediction),2)
        except ValueError:
            return "Please Enter valid values"
    return render_template("results.html",prediction=model_prediction)
    

@app.route('/predict/group',methods=["GET"])
def group():
    return render_template("group.html")

@app.route('/results/group',methods=["GET","POST"])
def result_group():
    df_test=pd.read_csv(request.files.get("file"),encoding='UTF-8')
    model = joblib.load(pickel)
    prediction=model.predict(df_test)
    print(str(list(prediction)))
    return render_template("resultsGroup.html",result=prediction)

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')