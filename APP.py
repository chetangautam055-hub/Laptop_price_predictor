"""
Laptop Price Predictor - FastAPI service
-----------------------------------------
Loads pipe.pkl (the trained sklearn Pipeline) and df.pkl (the cleaned
training dataframe, used only to expose valid dropdown values) and serves
predictions over HTTP.

Run with:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# 1. Load the pickled artifacts (must sit in the same folder as app.py)
# ---------------------------------------------------------------------
try:
    pipe = pickle.load(open("pipe.pkl", "rb"))
    df = pickle.load(open("df.pkl", "rb"))
except FileNotFoundError as e:
    raise RuntimeError(
        "pipe.pkl / df.pkl not found. Run the pickle.dump cells in your "
        "notebook and place both files next to app.py."
    ) from e

app = FastAPI(title="Laptop Price Predictor API")

# Allow calls from a browser-based frontend (React, Streamlit, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# 2. Define the request schema (what the user/frontend sends us)
# ---------------------------------------------------------------------
class LaptopSpecs(BaseModel):
    company: str = Field(..., example="Dell")
    type_name: str = Field(..., example="Ultrabook")
    ram: int = Field(..., example=8, description="RAM in GB")
    weight: float = Field(..., example=1.5, description="Weight in kg")
    touchscreen: bool = Field(..., example=False)
    ips: bool = Field(..., example=True)
    screen_size: float = Field(..., example=15.6, description="Screen size in inches")
    resolution: str = Field(..., example="1920x1080", description="e.g. 1920x1080")
    cpu_brand: str = Field(..., example="Intel Core i5")
    hdd: int = Field(..., example=0, description="HDD size in GB (0 if none)")
    ssd: int = Field(..., example=256, description="SSD size in GB (0 if none)")
    gpu_brand: str = Field(..., example="Intel")
    os: str = Field(..., example="Windows")


# ---------------------------------------------------------------------
# 3. Helper: turn the request into the exact row shape pipe.predict wants
#    Order MUST match training: Company, TypeName, Ram, Weight,
#    Touchscreen, IPS, ppi, Cpu brand, HDD, SSD, Gpu Brand, os
# ---------------------------------------------------------------------
def build_feature_row(specs: LaptopSpecs) -> pd.DataFrame:
    try:
        x_res, y_res = map(int, specs.resolution.lower().split("x"))
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="resolution must look like '1920x1080'",
        )

    ppi = ((x_res**2 + y_res**2) ** 0.5) / specs.screen_size

    row = pd.DataFrame(
        [[
            specs.company,
            specs.type_name,
            specs.ram,
            specs.weight,
            1 if specs.touchscreen else 0,
            1 if specs.ips else 0,
            ppi,
            specs.cpu_brand,
            specs.hdd,
            specs.ssd,
            specs.gpu_brand,
            specs.os,
        ]],
        columns=[
            "Company", "TypeName", "Ram", "Weight", "Touchscreen",
            "IPS", "ppi", "Cpu brand", "HDD", "SSD", "Gpu Brand", "os",
        ],
    )
    return row


# ---------------------------------------------------------------------
# 4. Routes
# ---------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Laptop Price Predictor API is running. See /docs"}


@app.get("/options")
def get_options():
    """Valid categorical values, read from the training data, so a
    frontend can populate dropdowns instead of hardcoding them."""
    return {
        "company": sorted(df["Company"].unique().tolist()),
        "type_name": sorted(df["TypeName"].unique().tolist()),
        "cpu_brand": sorted(df["Cpu brand"].unique().tolist()),
        "gpu_brand": sorted(df["Gpu Brand"].unique().tolist()),
        "os": sorted(df["os"].unique().tolist()),
        "ram_options": sorted(df["Ram"].unique().tolist()),
    }


@app.post("/predict")
def predict_price(specs: LaptopSpecs):
    row = build_feature_row(specs)
    log_price = pipe.predict(row)[0]        # model was trained on np.log(Price)
    price = float(np.exp(log_price))        # undo the log transform
    return {
        "predicted_price": round(price, 2),
        "currency": "INR"  # change if your Price column wasn't in rupees
    }
