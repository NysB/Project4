# import libraris
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle

## col names for df
col_names = ['ApplicantIncome', 
             'CoapplicantIncome', 
             'LoanAmount',
             'Loan_Amount_Term', 
             'Credit_History', 
             'Loan_Status', 
             'Gender_Female',
             'Gender_Male', 
             'Married_No', 
             'Married_Yes', 
             'Dependents_0',
             'Dependents_1', 
             'Dependents_2', 
             'Dependents_3+', 
             'Education_Graduate',
             'Education_Not Graduate', 
             'Self_Employed_No', 
             'Self_Employed_Yes',
             'Property_Area_Rural', 
             'Property_Area_Semiurban',
             'Property_Area_Urban']


# initialzie the flask app
app = Flask(__name__)

# load ml model
model = pickle.load(open('logistic_regression.pkl', 'rb'))

# define the app route for the default page of the web-app
@app.route('/')
def home():
    return render_template('index.html')

# define the app route for the default page of the web-app
@app.route('/predict', methods=['POST'])
def predict():

    X = np.zeros( len(col_names) )
    print("X",X)
    df_XX = pd.DataFrame(data=[dict(zip(col_names, X) ) ] )

    # for rending result on html GUI

    feat_ApplicantIncome = int( request.form['ApplicantIncome'] )
    print( "ApplicantIncome:", feat_ApplicantIncome )
    df_XX['ApplicantIncome'] = feat_ApplicantIncome

    feat_CoapplicantIncome = int( request.form['CoapplicantIncome'] )
    print( "CoapplicantIncome:", feat_CoapplicantIncome )
    df_XX['CoapplicantIncome'] = feat_CoapplicantIncome

    feat_loanAmount = int( request.form['LoanAmount'] )
    print( "LoanAmount:", feat_loanAmount )
    df_XX['LoanAmount'] = feat_loanAmount 

    feat_loan_amount_term = int( request.form['Loan_Amount_Term'] )
    print( "Loan_Amount_Term:", feat_loan_amount_term )
    df_XX['Loan_Amount_Term'] = feat_loan_amount_term

    feat_credit_history = int( request.form['Credit_History'] )
    print( "Credit_History:", feat_credit_history )
    df_XX['Credit_History'] = feat_credit_history

    feat_loan_Status = int( request.form['Loan_Status'] )
    print( "Loan_Status:", feat_loan_Status )
    df_XX['Loan_Status'] = feat_loan_Status

    feat_gender = int(request.form ['Gender'])
    print ("gender:",feat_gender)
    df_XX[feat_gender] = 1.0

    feat_married = int(request.form ['Married'])
    print ("married:",feat_married)
    df_XX[feat_married] = 1.0

    feat_dependents = int(request.form ['Dependents'])
    print ("dependents:",feat_dependents)
    df_XX[feat_dependents] = 1.0
    
    feat_education = int(request.form ['Education'])
    print ("education:",feat_education)
    df_XX[feat_education] = 1.0

    feat_self_employed = int(request.form ['Self_Employed'])
    print ("self_employed:",feat_self_employed)
    df_XX[feat_self_employed] = 1.0

    feat_property_area = int(request.form ['Property_Area'])
    print ("property_area:",feat_property_area)
    df_XX[feat_property_area] = 1.0

    prediction = model.predict_proba(df_XX)
    print("prediction",prediction)

    output = np.round(prediction[0][1], 2)
    print('Your Loan Status is: {}'.format(output))
    

    return render_template('index.html', prediction_text='Your Loan Status is: {}'.format(output))
    

# start the flask server
if __name__ == '__main__':
    app.run(debug=True)