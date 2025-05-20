from fastapi import FastAPI

import pickle
import pandas as pd

from pydantic import BaseModel

class Water(BaseModel):
    ph : float
    Hardness : float
    Solids : float
    Chloramines : float
    Sulfate : float
    Conductivity : float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity : float

app = FastAPI(
    title="Water Potability Prediction",
    description="To predict is it safe to drink water"
    )

with open("../model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def index():
    return "Welcome to Water Potability Prediction app"

@app.post("/predict")
def model_predict(water: Water):
    sample = pd.DataFrame({
        'ph' : [water.ph],
        'Hardness' : [water.Hardness],
        'Solids' :[water.Solids],
        'Chloramines' :[water.Chloramines],
        'Sulfate' : [water.Sulfate],
        'Conductivity' :[water.Conductivity],
        'Organic_carbon' :[water.Organic_carbon],
        'Trihalomethanes' : [water.Trihalomethanes],
        'Turbidity' :[water.Turbidity]
    })

    predicted_value = model.predict(sample)


    if predicted_value == 1:
        return "Water is Consumable"
    else:
        return "Water is not Consumable"