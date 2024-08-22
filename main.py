from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
from Model.model import load_model  


app = FastAPI()


model = load_model()


class PredictionRequest(BaseModel):
    previous_fare: float
    nsmiles: float
    passengers: int
    quarter: int


@app.get("/")
def read_root():
    return {"message": "Flight Fare Prediction API"}

@app.post("/predict/")
def predict(request: PredictionRequest):
    try:

        input_data = pd.DataFrame([request.dict()])


        prediction = model.predict(input_data)
        
        prediction = prediction[0]+request.previous_fare

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
        
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8080)