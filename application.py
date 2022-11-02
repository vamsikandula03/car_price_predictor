import pickle

import pandas as pd
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
car = pd.read_csv("cleaneddataset.csv")

model = pickle.load(open("LinearRegressionmodel.pkl", "rb"))


@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    return render_template('index1.html', companies=companies, car_models=car_models, years=year, fuel_type=fuel_type,
                           gcompany="", gyear="", gcarmodel="", gfuel="",gkmsdriven="")


@app.route('/predict', methods=['POST'])
def predict():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    company = request.form.get('companyname')
    carmodel = request.form.get('companybrand')
    year = int(request.form.get('yearsel'))
    fuelty = request.form.get('fuelsel')
    kmsdriven = int(request.form.get('kmgiven'))
    prediction = model.predict(pd.DataFrame([[carmodel, company, year, kmsdriven, fuelty]],
                                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    return render_template("index1.html", prediction_text="Predicted Price is ₹ " + str(int(prediction)),
                           companies=companies, gcompany=company, car_models=car_models, gcarmodel=carmodel,
                           years=years, gyear=year, fuel_type=fuel_type, gfuel=fuelty,gkmsdriven=kmsdriven)


if __name__ == "__main__":
    app.run(debug=True)
