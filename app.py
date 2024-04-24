from flask import Flask, render_template
import pandas as pd
import plotly.graph_objects as go
from pmdarima.arima import auto_arima
import numpy as np

app = Flask(__name__)

# Load the data from CSV
covid = pd.read_csv("covid_19_data.csv")
covid.drop(["SNo"], axis=1, inplace=True)
covid["ObservationDate"] = pd.to_datetime(covid["ObservationDate"])
grouped_country = covid.groupby(["Country/Region", "ObservationDate"]).agg({"Confirmed": 'sum',
                                                                           "Recovered": 'sum',
                                                                           "Deaths": 'sum'})
grouped_country["Active Cases"] = grouped_country["Confirmed"] - grouped_country["Recovered"] - grouped_country[
    "Deaths"]
grouped_country["log_confirmed"] = np.log(grouped_country["Confirmed"])
grouped_country["log_active"] = np.log(grouped_country["Active Cases"])
datewise = covid.groupby(["ObservationDate"]).agg({"Confirmed": 'sum', "Recovered": 'sum', "Deaths": 'sum'})
datewise["Days Since"] = datewise.index - datewise.index.min()

model_train = datewise.iloc[:int(datewise.shape[0] * 0.95)]
valid = datewise.iloc[int(datewise.shape[0] * 0.95):]
y_pred = valid.copy()

def predict_cases(train_data, valid_data):
    model_ma = auto_arima(train_data, trace=True, error_action='ignore', start_p=0, start_q=0, max_p=0,
                      max_q=2, suppress_warnings=True, stepwise=False, seasonal=False)
    model_ma.fit(train_data)
    prediction_ma = model_ma.predict(len(valid_data))
    return prediction_ma

y_pred["Confirmed_Prediction"] = predict_cases(model_train["Confirmed"], valid["Confirmed"])
y_pred["Recovered_Prediction"] = predict_cases(model_train["Recovered"], valid["Recovered"])
y_pred["Deaths_Prediction"] = predict_cases(model_train["Deaths"], valid["Deaths"])

@app.route('/')
def index():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Confirmed"], mode='lines+markers', name="Confirmed Cases"))
    fig.add_trace(go.Scatter(x=valid.index, y=y_pred["Confirmed_Prediction"], mode='lines+markers',
                             name="Confirmed Cases Prediction"))
    fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Recovered"], mode='lines+markers', name="Recovered Cases"))
    fig.add_trace(go.Scatter(x=valid.index, y=y_pred["Recovered_Prediction"], mode='lines+markers',
                             name="Recovered Cases Prediction"))
    fig.add_trace(go.Scatter(x=datewise.index, y=datewise["Deaths"], mode='lines+markers', name="Deaths Cases"))
    fig.add_trace(go.Scatter(x=valid.index, y=y_pred["Deaths_Prediction"], mode='lines+markers',
                             name="Deaths Cases Prediction"))
    fig.update_layout(title="COVID-19 Cases and Forecasting",
                      xaxis_title="Date", yaxis_title="Number of Cases")
    plot_div = fig.to_html(full_html=False)
    return render_template('index.html', plot=plot_div)

if __name__ == "__main__":
    app.run(debug=True)
