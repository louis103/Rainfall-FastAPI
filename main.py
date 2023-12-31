from fastapi import FastAPI, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import requests
from io import StringIO
import datetime
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import statsmodels.api as sm
import json
from fastapi.responses import JSONResponse


# Initialize the fastAPI
app = FastAPI(
    title="Weather API",
    description="An Api that generates rainfall forecasts based on user location",
    version="0.1.0",
    openapi_url="/api/v0.1.1/openapi.json",
)

# configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# get today's date so that we adjust the API
def getToday():
    today_date = datetime.date.today()
    formatted_date = today_date.strftime("%Y-%m-%d")
    formatted_date = formatted_date.replace("-", "")
    return formatted_date


# initialize today's date
today = getToday()  # 20230913

# Define the NASA Power API URL template with placeholders for latitude and longitude
NASA_POWER_API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point?start=20130101&end={today}&latitude={latitude}&longitude={longitude}&parameters=PRECTOTCORR&community=AG&header=false&format=csv"


# T2M,PS,WS10M,QV2M,
# Home route
@app.get("/")
async def home():
    return {"message": "Welcome to Rainfall Forecasting API!"}


# Main API route -> http://localhost:8000/get_forecast/?latitude=-1.2345&longitude=36.67845
# Production URL -> https://rainfall-forecasting-api.azurewebsites.net/get_forecast/?latitude=-1.2345&longitude=36.67845
@app.get("/get_forecast/")
async def get_coordinates(
    latitude: float = Query(..., description="Latitude"),
    longitude: float = Query(..., description="Longitude"),
):
    dataframe = get_csv_by_coordinates(latitude, longitude)
    # run the forecasts
    forecast_data = run_model(dataframe, latitude, longitude, n_forecasts=28)
    return forecast_data


# Function to download CSV data by lat:-0.091702 and lon:34.767956
def get_csv_by_coordinates(latitude: float, longitude: float):
    api_url = NASA_POWER_API_URL.format(
        latitude=latitude, longitude=longitude, today=today
    )
    response = requests.get(api_url)
    response.raise_for_status()
    csv_data = response.text
    df = pd.read_csv(
        StringIO(csv_data),
        skiprows=0,
        parse_dates={"date": ["YEAR", "DOY"]},
        date_parser=parse_date,
        skipinitialspace=True,
        index_col=0,
    )
    # rename columns"T2M": "Temp2M",
    # "PS": "SurfacePressure",
    # "WS10M": "windspeed10M",
    # "QV2M": "Humidity2M",
    df.rename(
        columns={
            "PRECTOTCORR": "precipitation",
        },
        inplace=True,
    )
    # drop any null values
    df.drop(df.index[df["precipitation"] == -999.00], inplace=True)
    # select data from 2013 January
    start_date = "2013-01-01"
    df = df[start_date:]
    # Set the frequency of the DataFrame
    df = df.set_index(pd.date_range(start=start_date, periods=len(df), freq="D"))
    df = df["precipitation"]
    return df


# Parse the dates from CSV file
def parse_date(x):
    return dt.strptime(x, "%Y %j")


# check dataframe stationarity
def is_stationary(dataframe):
    """
    Receives a pandas dataframe,
    Checks for stationarity in data: This is useful for ARIMA
    return either True or False
    """
    adf_pvalue = adfuller(dataframe)[1]
    kpss_pvalue = kpss(dataframe)[1]
    if adf_pvalue < 0.05 and kpss_pvalue > 0.05:
        return True
    elif adf_pvalue >= 0.05:
        return False
    else:
        return False


# check for seasonality in dataframe
def has_seasonality(dataframe):
    decomposition = sm.tsa.seasonal_decompose(dataframe, model="additive")
    seasonal_component = decomposition.seasonal
    seasonal_mean = np.mean(seasonal_component)
    seasonal_threshold = 0.1

    return (
        abs(seasonal_mean) > seasonal_threshold
    )  # return a True: seasonal, False: non-seasonal


# Using the prepared data, apply ARIMA and return 28 precipitation day forecasts
def run_model(dataframe, latitude, longitude, n_forecasts=28):
    """
    Receive the processed dataframe,
    feed it to ARIMA,
    make 30-day precipitation forecasts for the location
    return a JSON reponse for both 30 day forecast and weekly forecast
    """
    # Fit the model
    best_model = get_arima_order(dataframe)  # returns something like (2,1,3)
    model = ARIMA(dataframe, order=best_model).fit()

    # get the future 28 days for forecasting and convert them to JSON.
    forecast_values = model.forecast(steps=n_forecasts)
    # get the weekly analysis from the forecasts
    weekly_analysis = analyze_weekly_forecasts(forecast_values)
    forecast_json = forecast_values.to_json(date_format="iso", orient="split")
    parsed_json = json.loads(forecast_json)

    # Process the dates to remove milliseconds and store in a new list
    dates = [date.split("T")[0] for date in parsed_json["index"]]
    forecast_data_values = [round(x, 2) for x in parsed_json["data"]]

    # Create a dictionary in the desired format
    forecast_dict = {
        "daily_forecasts": {"date": dates, "forecasts": forecast_data_values},
        "weekly_analysis": weekly_analysis,
        "coordinates": [latitude, longitude],
    }

    return JSONResponse(forecast_dict, status_code=status.HTTP_200_OK)


def get_arima_order(df):
    is_seasonal = has_seasonality(df)
    if is_seasonal:  # we have seasonality in data #stationary="adfuller"
        arima_model = auto_arima(df, seasonal=True)  # test="adf"
    else:
        arima_model = auto_arima(df, seasonal=False)

    # return the best ARIMA parameters to fit the data
    return arima_model.order


# function to give us 30 dates in the future
def generate_future_dates(df):
    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=30, freq="D")
    return forecast_dates


# get weekly analysis from the forecasts
def analyze_weekly_forecasts(forecast_series):
    # Initialize the weekly analysis dictionary
    weekly_analysis = {}
    total_monthly_rainfall = 0

    # Calculate the number of weeks (assuming 7 days per week)
    num_weeks = len(forecast_series) // 7

    for week_number in range(1, num_weeks + 1):
        week_start = forecast_series.index[7 * (week_number - 1)]
        week_end = forecast_series.index[7 * week_number - 1]

        # Calculate the total precipitation for the week
        total_precipitation = round(forecast_series[week_start:week_end].sum(), 2)

        # Calculate the mean daily precipitation for the week
        mean_daily_precipitation = round(forecast_series[week_start:week_end].mean(), 2)

        total_monthly_rainfall += total_precipitation

        # Determine the recommendation based on the comparison to the 28-day mean
        # Determine the recommendation based on the comparison to different ranges
        if mean_daily_precipitation < 4:
            recommendation = "Very Very Tiny Rains"
        elif 4 <= mean_daily_precipitation < 10:
            recommendation = "Very Tiny Rains"
        elif 10 <= mean_daily_precipitation < 15:
            recommendation = "Tiny Rains"
        elif 15 <= mean_daily_precipitation < 20:
            recommendation = "Light Rains"
        elif 20 <= mean_daily_precipitation < 30:
            recommendation = "Moderate Rains"
        elif 30 <= mean_daily_precipitation < 60:
            recommendation = "Heavy Rains"
        else:
            recommendation = "Very Very Heavy Rains"

        week_name = f"Week {week_number} {week_start.strftime('%Y-%m-%d')}"
        week_data = {
            "total_precipitation": round(total_precipitation, 2),
            "mean_daily_precipitation": round(mean_daily_precipitation, 2),
            "recommendation": recommendation,
        }

        weekly_analysis[week_name] = week_data
    weekly_analysis["cummulative_sum"] = total_monthly_rainfall

    return weekly_analysis
