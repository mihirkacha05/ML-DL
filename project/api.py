from fastapi import FastAPI
from pydantic import BaseModel,Field,model_validator
import pickle
import pandas as pd

app = FastAPI()

model = pickle.load(open("cardio_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

features = ['age_years','gender','height','weight','bmi',
            'ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']


class InputData(BaseModel):
    age_years: int = Field(..., ge=18, le=100)
    gender: int = Field(..., ge=0, le=1)
    height: int = Field(..., ge=120, le=220)
    weight: float = Field(..., ge=30, le=200)

    ap_hi: int = Field(..., ge=90, le=250)
    ap_lo: int = Field(..., ge=60, le=150)

    cholesterol: int = Field(..., ge=1, le=3)
    gluc: int = Field(..., ge=1, le=3)
    smoke: int = Field(..., ge=0, le=1)
    alco: int = Field(..., ge=0, le=1)
    active: int = Field(..., ge=0, le=1)

    @model_validator(mode="after")
    def check_bp(self):
        if self.ap_hi <= self.ap_lo:
            raise ValueError(
                "Systolic BP (ap_hi) must be greater than Diastolic BP (ap_lo)"
            )
        return self



@app.post("/predict")
def predict(data: InputData):
    bmi = data.weight / (data.height/100) ** 2

    df = pd.DataFrame([[ 
        data.age_years, data.gender, data.height, data.weight,
        bmi, data.ap_hi, data.ap_lo, data.cholesterol,
        data.gluc, data.smoke, data.alco, data.active
    ]], columns=features)

    scaled = scaler.transform(df)
    pred = int(model.predict(scaled)[0])
    prob = float(model.predict_proba(scaled)[0][1])

    return {"prediction": pred, "probability": round(prob, 2)}
