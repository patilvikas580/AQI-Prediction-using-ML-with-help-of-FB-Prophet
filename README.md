# AQI-Prediction-using-ML-with-help-of-FB-Prophet
## Project Workflow
**Loading Data:**
<br>
- The air quality data is loaded into a Pandas DataFrame from a CSV file.
**Data Cleaning:**

- The data is initially not in the correct format. Adjustments are made to handle delimiters and decimal points.
**Unnecessary columns are removed.**
- Missing values, represented by -200, are replaced with NaN and subsequently filled with the mean of their respective columns.
**Data Preparation:**

- Date and time columns are converted to appropriate formats and combined into a single column.
- The dataset is prepared to match the requirements of the Prophet forecasting model.
## Forecasting:
- The Prophet model is trained using the cleaned dataset.
- Future air quality trends are predicted using the trained model.
- The results are visualized to understand future trends and seasonal components.
## Steps in Detail
**Loading Data**
- The data is loaded using Pandas' read_csv function. Proper delimiters and decimal separators are specified to ensure correct data formatting.

**python**
- air_quality_data = pd.read_csv('/content/AirQualityUCI.csv', sep=';', decimal=',')
**Data Cleaning**
- Unnecessary columns are removed, and missing values are handled:

 **code**
air_quality_data = air_quality_data.iloc[:, :-2]
air_quality_data = air_quality_data.replace(to_replace=-200, value=np.NaN)
air_quality_data = air_quality_data.fillna(air_quality_data.mean())
Data Preparation
Date and time columns are combined, and a new DataFrame is created for the Prophet model:

 **code**
date_info = pd.to_datetime(air_quality_data['Date'])
time_info = air_quality_data['Time'].apply(lambda x: x.replace('.', ':'))
date_time = pd.concat([date_info, time_info], axis=1)
date_time['ds'] = date_time['Date'].astype(str) + ' ' + date_time['Time'].astype(str)

data = pd.DataFrame()
data['ds'] = pd.to_datetime(date_time['ds'], format="%Y-%m-%d %H:%M:%S")
data['y'] = air_quality_data['RH']
## Forecasting
**The Prophet model is trained and used to predict future trends:**
**code**
from prophet import Prophet

model = Prophet()
model.fit(data)

future = model.make_future_dataframe(periods=365, freq='H')
forecast = model.predict(future)
**Visualization**
The forecast results are visualized to understand future trends:
**code**
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)
## Conclusion
-This project demonstrates the complete workflow of data cleaning, preparation, and forecasting using a real-world air quality dataset. The Prophet library is used for time series forecasting, providing insights into future trends and seasonal patterns in the data.

## Dependencies
- Python 3.x
- Pandas
- Numpy
- Prophet
**Usage**
- Clone the repository.
- Ensure you have the necessary dependencies installed.
- Run the script to see the data analysis and forecasting results.
## License
This project is licensed under the MIT License. Feel free to use and modify it as per your needs.

