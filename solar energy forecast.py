# Importing Package
from functions import solar_power_forecast as spf

# User Inputs
api_key = ''
plant_capacity = 100  # MW
lat, lon = 40.7128, -74.0060  # Example: New York City

# Get Results
forecast,optimal_tilt_angle = spf(lat, lon, plant_capacity, api_key)
