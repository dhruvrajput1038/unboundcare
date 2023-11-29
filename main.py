from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import joblib
# import sklearn.linear_model
# from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__,template_folder='templates')



@app.route('/')
def hello_world():
    return render_template("index.html")
    # return render_template("index.html")

@app.route('/diabetes/diabetes.html')
def diabetes():
    return render_template("/diabetes/diabetes.html")

@app.route('/diabetes/predict.html', methods=['POST', 'GET'])
def dia_predict():
    pregnancies = request.form['pregnancies']
    glucose = request.form['glucose']
    blood_pressure = request.form['blood-pressure']
    skin_thickness = request.form['skin-thickness']
    insulin = request.form['insulin']
    bmi = request.form['bmi']
    pedigree = request.form['pedigree']
    age = request.form['age']
    
    loaded_model = pickle.load(open(r'C:\Users\MI\OneDrive\Desktop\Unbound\Dashboard\models/diabetes_model.sav', 'rb'))
    input_data = [pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,pedigree,age]
    print(input_data)
    # input_data = (5,166,72,19,175,25.8,0.587,51)
    # text = [value for key, value in request.form.items()]
    input_data = np.array(input_data, dtype=float).reshape(1, -1)
    prediction = loaded_model.predict(input_data)


    if prediction[0] == 0:
        return render_template('/diabetes/predict.html', output='The person is NOT diabetic')
    else:
        return render_template('/diabetes/predict.html', output='The person is diabetic')


    
@app.route('/heart/heart.html')
def heart():
    return render_template("/heart/heart.html")
    
    
@app.route('/heart/predict.html', methods=['POST', 'GET'])
def heart_predict():
    loaded_model_heart = pickle.load(open(r'C:\Users\MI\OneDrive\Desktop\Unbound\Dashboard\models/heart_disease_model.sav', 'rb'))
    # input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
    # text = [value for key, value in request.form.items()]
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    thestbps = int(request.form['thestbps'])
    chol = int(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = int(request.form['slope'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])
    
    input_data = [age,sex,cp,thestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
    # if not text:
    #     # return render_template('/heart/predict.html', output='Invalid input data')
    #     return render_template('/heart/predict.html', output=text)

    # change the input data to a numpy array
    input_data_as_numpy_array= np.asarray(input_data)

    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    print(input_data_reshaped)
    prediction = loaded_model_heart.predict(input_data_reshaped)
    print(prediction)
    # print(prediction)

    if (prediction[0]== 0):
        # print('The Person does not have a Heart Disease')
        return render_template('/heart/predict.html', output='The person is  Heart Disease😔😔')
        # return render_template('/result/Pos_result.html', output='The person is NOT Heart Disease😁😁')
    else:
        return render_template('/heart/predict.html', output='The person is  Heart Disease😔😔')
        # return render_template('/result/Neg_result.html', output='The person is  Heart Disease😔😔')
        # print('The Person has Heart Disease')







@app.route('/parkinson/parkinson.html')
def parkinson():
    return render_template("/parkinson/parkinson.html")



@app.route('/parkinson/predict.html', methods=['POST', 'GET'])
def parkinson_predict():
    
    loaded_model_parkinson = pickle.load(open(r'C:\Users\MI\OneDrive\Desktop\Unbound\Dashboard\models/parkinsons_model.sav', 'rb'))
    # input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)


    name = (request.form['name'])
    MDVP_Fo_Hz_ = float(request.form['MDVP:Fo(Hz)'])
    MDVP_Fhi_Hz_ = float(request.form['MDVP:Fhi(Hz)'])
    MDVP_Flo_Hz_ = float(request.form['MDVP:Flo(Hz)'])
    MDVP_Jitter_percentage_ = float(request.form['MDVP:Jitter(%)'])
    MDVP_Jitter_abs_ = float(request.form['MDVP:Jitter(Abs)'])
    MDVP_RAP = float(request.form['MDVP:RAP'])
    MDVP_PPQ  = float(request.form['MDVP:PPQ'])
    Jitter_DDP = float(request.form['Jitter:DDP'])
    MDVP_Shimmer = float(request.form['MDVP:Shimmer'])
    MDVP_Shimmer_dB_ = float(request.form['MDVP:Shimmer(dB)'])
    Shimmer_APQ3  = float(request.form['Shimmer:APQ3'])
    Shimmer_APQ5 = float(request.form['Shimmer:APQ5'])
    MDVP_APQ = float(request.form['MDVP:APQ'])
    Shimmer_DDA = float(request.form['Shimmer:DDA'])
    NHR = float(request.form['NHR'])
    HNR = float(request.form['HNR'])
    status = int(request.form['status'])
    RPDE = float(request.form['RPDE'])
    D2 = float(request.form['D2'])
    DFA = float(request.form['DFA'])
    spread1 = float(request.form['spread1'])
    spread2 = float(request.form['spread2'])
    PPE = float(request.form['PPE'])
    
    
    
    input_data = [MDVP_Fo_Hz_,MDVP_Fhi_Hz_,MDVP_Flo_Hz_,MDVP_Jitter_percentage_,MDVP_Jitter_abs_,MDVP_RAP,MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_Shimmer_dB_,Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]
    # changing input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    
    # reshape the numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the data
    # scaler = StandardScaler()
    scaler = joblib.load(r'C:\Users\MI\OneDrive\Desktop\Unbound\Dashboard\models/scaler_filename.pkl')
    # scaler = pickle.load(open(r'C:\Users\MI\OneDrive\Desktop\Unbound\Dashboard\models/scaler_filename.pkl', 'rb'))
    std_data = scaler.transform(input_data_reshaped)

    prediction = loaded_model_parkinson.predict(std_data)
    print(prediction)


    if (prediction[0] == 0):
        # print("The Person does not have Parkinsons Disease")
        return render_template('/parkinson/predict.html', output='The person has NOT Parkinson Disease😁😁')
    else:
        # print("The Person has Parkinsons")
        return render_template('/parkinson/predict.html', output='The person has Parkinson Disease😔😔😔')





    # # input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
    # # text = [value for key, value in request.form.items()]
    # age = int(request.form['age'])
    # sex = int(request.form['sex'])
    # cp = int(request.form['cp'])
    # thestbps = int(request.form['thestbps'])
    # chol = int(request.form['chol'])
    # fbs = int(request.form['fbs'])
    # restecg = int(request.form['restecg'])
    # thalach = int(request.form['thalach'])
    # exang = int(request.form['exang'])
    # oldpeak = float(request.form['oldpeak'])
    # slope = int(request.form['slope'])
    # ca = int(request.form['ca'])
    # thal = int(request.form['thal'])
    
    # input_data = [age,sex,cp,thestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
    # # if not text:
    # #     # return render_template('/heart/predict.html', output='Invalid input data')
    # #     return render_template('/heart/predict.html', output=text)

    # # change the input data to a numpy array
    # input_data_as_numpy_array= np.asarray(input_data)

    # # reshape the numpy array as we are predicting for only on instance
    # input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    # print(input_data_reshaped)
    # prediction = loaded_model_heart.predict(input_data_reshaped)
    # print(prediction)
    # # print(prediction)

    # if (prediction[0]== 0):
    #     # print('The Person does not have a Heart Disease')
    #     return render_template('/heart/predict.html', output='The person is NOT Heart Disease😁😁')
    # else:
    #     return render_template('/heart/predict.html', output='The person is  Heart Disease😔😔')
    #     # print('The Person has Heart Disease')
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    app.run(debug=True)
