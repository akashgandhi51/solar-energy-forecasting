import math
import requests
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from geopy.geocoders import Photon
from datetime import datetime, timedelta

def fetch_weather_forecast(lat, lon, api_key):
    """
    Fetches the 5-day weather forecast from the OpenWeatherMap API for a given latitude and longitude.

    Parameters:
    - lat (float): Latitude of the location.
    - lon (float): Longitude of the location.
    - api_key (str): API key for accessing the OpenWeatherMap API.

    Returns:
    - list: A list of weather forecast data points, each as a dictionary. Returns an empty list if the request fails.
    """
    url = f'https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric'
    response = requests.get(url)
    if response.status_code == 200:
        weather_forecast = response.json()
        return weather_forecast['list']
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return []
        
def prepare_weather_data(weather_forecast_data):
    """
    Prepares the weather forecast data for further processing and modeling. Converts the forecast data into an hourly format, 
    handles missing values through interpolation, and generates additional features like month, hour, and temperature differences.

    Parameters:
    - weather_forecast_data (list): List of weather forecast data points obtained from the OpenWeatherMap API.

    Returns:
    - DataFrame: A pandas DataFrame containing the prepared and interpolated hourly forecast data with additional features.
    """
    # Convert to a DataFrame
    forecast_df = pd.DataFrame([{'datetime': item['dt_txt'],'temp': item['main']['temp']} for item in weather_forecast_data])
    # Convert 'datetime' from string to datetime object
    forecast_df['datetime'] = pd.to_datetime(forecast_df['datetime'])
    # Interpolate to get hourly forecasts
    hourly_forecast_df = forecast_df.set_index('datetime').resample('H').interpolate()
    # Reset index to make 'datetime' a column again
    hourly_forecast_df.reset_index(inplace=True)
    # Convert 'datetime' from UTC to U.S. Central Time (UTC-5)
    hourly_forecast_df['datetime'] = hourly_forecast_df['datetime'] - timedelta(hours=5)
    
    # Ensure the DataFrame covers a full 7-day period for hourly forecasts
    start_date = hourly_forecast_df['datetime'].min()
    end_date = start_date + timedelta(days=8)
    # Creating a full range of hourly timestamps from start to end
    full_range = pd.date_range(start=start_date, end=end_date, freq='H')
    forecast_df_full = hourly_forecast_df.set_index('datetime').reindex(full_range).reset_index().rename(columns={'index': 'datetime'})
    forecast_df_full = forecast_df_full[forecast_df_full['datetime'] < datetime.strftime(end_date.date(),'%Y-%m-%d')].copy()
    
    # Preparing empty DataFrame for interpolated hourly data
    interpolated_df = pd.DataFrame()
    # Segment data by each hour and interpolate within each segment
    for hour in range(24):
        # Filter rows for the specific hour across all days
        hourly_data = forecast_df_full[forecast_df_full['datetime'].dt.hour == hour].copy()
            
        # Linear interpolation for the current hour's data
        hourly_data.interpolate(method='linear', inplace=True)
        
        # Handling missing data at the start or end with backfill and forward fill
        hourly_data.fillna(method='bfill', inplace=True)
        hourly_data.fillna(method='ffill', inplace=True)

        # Append the interpolated data for the current hour to the full DataFrame
        interpolated_df = pd.concat([interpolated_df, hourly_data], axis=0)

    # Sort the full DataFrame by datetime to ensure chronological order
    interpolated_df.sort_values(by='datetime', inplace=True)
    
    # Prepare additional features required for the model, such as 'Month', 'Hour', etc.
    interpolated_df['Month'] = interpolated_df['datetime'].dt.month
    interpolated_df['Hour'] = interpolated_df['datetime'].dt.hour
    interpolated_df['Day'] = interpolated_df['datetime'].dt.day
    interpolated_df['Year'] = interpolated_df['datetime'].dt.year
    
    # Calculating daily max, min, and average temperature
    daily_stats = interpolated_df.groupby(['Year', 'Month', 'Day']).agg(
        max_temp=pd.NamedAgg(column='temp', aggfunc='max'),
        min_temp=pd.NamedAgg(column='temp', aggfunc='min'),
        avg_temp=pd.NamedAgg(column='temp', aggfunc='mean')
    ).reset_index()
    
    # Prepare features for the model
    interpolated_df = pd.merge(interpolated_df, daily_stats, on=['Year', 'Month', 'Day'])
    # Creating a temperature difference feature
    interpolated_df['temp_diff'] = interpolated_df['max_temp'] - interpolated_df['min_temp']
    hourly_forecast_df = interpolated_df[interpolated_df['datetime'] >= datetime.strftime(start_date.date() + timedelta(days=1),'%Y-%m-%d')]
    hourly_forecast_df = hourly_forecast_df.reset_index(drop=True)
    return hourly_forecast_df

def predict_ghi(hourly_forecast_df):
    """
    Predicts Global Horizontal Irradiance (GHI) using a pre-trained LightGBM model and the prepared weather forecast data.

    Parameters:
    - hourly_forecast_df (DataFrame): A pandas DataFrame containing hourly weather forecast data and additional features.

    Returns:
    - DataFrame: The input DataFrame with an added column for the forecasted GHI values.
    
    The function loads a pre-trained LightGBM model and uses it to predict GHI based on the forecast data.
    """
    model_features_df = hourly_forecast_df[['Month', 'Hour', 'temp', 'max_temp', 'min_temp', 'avg_temp', 'temp_diff']].copy()
    gbm = lgb.Booster(model_file='model.txt')  # Assuming 'model.txt' is the pre-trained model file
    ghi_forecasts = gbm.predict(model_features_df)
    hourly_forecast_df['ghi_forecasted'] = ghi_forecasts
    return hourly_forecast_df

def calculate_max_ghi(lat, lon):
    """
    Calculates the maximum Global Horizontal Irradiance (GHI) for a given location using the NREL API.

    Parameters:
    - lat (float): Latitude of the location.
    - lon (float): Longitude of the location.

    Returns:
    - float: The maximum GHI value for the location, adjusted by adding the standard deviation to the maximum monthly average GHI.

    Makes an API call to NREL, processes the response to find the maximum GHI value, and adjusts this value by adding the standard deviation of the monthly average GHIs.
    """
    api_key = '1pi5lCtwT1yCRbHoe8XLViXFc5crsTQKno3qBnhI'  # Your API key here
    url = f'https://developer.nrel.gov/api/solar/solar_resource/v1.json?api_key={api_key}&lat={lat}&lon={lon}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        avg_ghi_df = pd.DataFrame(data['outputs']['avg_ghi']['monthly'], index=['avg_ghi']).T
        max_ghi = (avg_ghi_df['avg_ghi'].max() + avg_ghi_df['avg_ghi'].std()) * 100
        return max_ghi
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return 1000  # Default value in case of API failure
    
def calculate_power_generation(hourly_forecast_df, capacity, lat, lon):
    """
    Calculates the expected power generation based on forecasted GHI and the plant's capacity.

    Parameters:
    - hourly_forecast_df (DataFrame): DataFrame containing hourly forecasted GHI values.
    - capacity (float): The maximum capacity of the solar plant (in MW).
    - lat (float): Latitude of the solar plant.
    - lon (float): Longitude of the solar plant.

    Returns:
    - DataFrame: The input DataFrame augmented with a 'power' column representing the forecasted power generation for each hour.

    The function calculates the power generation by normalizing forecasted GHI values based on the maximum GHI and the plant's capacity. It also includes a visualization of average power generation by hour.
    """
    max_ghi = calculate_max_ghi(lat, lon)
    ratio = capacity / max_ghi
    hourly_forecast_df['power'] = hourly_forecast_df['ghi_forecasted'] * ratio
    hourly_forecast_df['power'] = hourly_forecast_df['power'].apply(lambda x: 0 if x < 0 else x)
    
    # Visualization of hourly power generation trends
    hourly_averages = hourly_forecast_df.groupby(['Hour']).mean()
    plt.figure(figsize=(14, 8))
    plt.plot(hourly_averages.index, hourly_averages['power'], marker='o', label='Power')
    plt.title('Hourly Averages of Power Generation')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Power Generation (MW)')
    plt.xticks(range(0, 24), [f'{hour}:00' for hour in range(0, 24)], rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()
    return hourly_forecast_df

def fetch_location_details(lat, lon):
    """
    Fetches location details such as state and country for a given latitude and longitude using the Photon geocoder.

    Parameters:
    - lat (float): Latitude of the location.
    - lon (float): Longitude of the location.

    Returns:
    - str: The country name of the given location.

    Utilizes the Photon API to reverse geocode the latitude and longitude, extracting and returning the country name from the response.
    """
    geolocator = Photon(user_agent="geoapiExercises")
    location = geolocator.reverse(f"{lat},{lon}")
    return location[0].split(',')[-1].strip()  # Extracting the country name

def calculate_optimal_tilt_angle(hourly_forecast_df, lat):
    """
    Calculates the optimal tilt angle for solar panels based on the latitude and the season.

    Parameters:
    - hourly_forecast_df (DataFrame): DataFrame containing the datetime of forecasts.
    - lat (float): Latitude of the location.

    Returns:
    - float: The optimal tilt angle forsolar panels at the given latitude.

    Utilizes the latitude and the specific day of the year to adjust the tilt angle for optimal sunlight exposure. The adjustment is based on general seasonal variations.

    """
    # Calculate day of the year from the forecast's datetime
    day_of_year = hourly_forecast_df.loc[0, 'datetime'].timetuple().tm_yday

    # Seasonal adjustments based on latitude and day of the year
    if day_of_year >= 79 and day_of_year <= 263:  # Spring to Fall in Northern Hemisphere
        optimal_angle = lat
    elif day_of_year >= 356 or day_of_year <= 79:  # Winter
        optimal_angle = lat + 15  # Adjust for winter, increase tilt
    else:  # Summer
        optimal_angle = max(lat - 15, 0)  # Adjust for summer, decrease tilt if positive

    print("Optimal Tilt Angle:", optimal_angle)
    return optimal_angle

def solar_power_forecast(lat, lon, plant_capacity, api_key):
    """
    Main function to forecast solar power generation and calculate the optimal tilt angle for solar panels for a specified location and plant capacity.

    Parameters:
    - lat (float): Latitude of the location.
    - lon (float): Longitude of the location.
    - plant_capacity (float): The maximum capacity of the solar power plant in MW.
    - api_key (str): API key for accessing weather data.

    Returns:
    - Tuple (DataFrame, float): A DataFrame with the hourly power generation forecast for 7 days and the optimal tilt angle for solar panels.

    This function integrates the workflow from fetching weather forecasts to calculating power generation and determining the optimal panel tilt angle.
    """
    # Ensure location is in the US for this implementation
    country = fetch_location_details(lat, lon)
    if country not in ["USA", "United States", "United States of America"]:
        print("This function currently works with US locations only.")
        return

    # Fetch and prepare weather data
    weather_forecast_data = fetch_weather_forecast(lat, lon, api_key)
    if weather_forecast_data:
        prepared_data = prepare_weather_data(weather_forecast_data)
        hourly_forecast_df = predict_ghi(prepared_data)
        hourly_forecast_df = calculate_power_generation(hourly_forecast_df, plant_capacity, lat, lon)
        optimal_tilt_angle = calculate_optimal_tilt_angle(hourly_forecast_df, lat)
        return hourly_forecast_df, optimal_tilt_angle
    else:
        print("Weather forecast data is not available.")
        return None
    
