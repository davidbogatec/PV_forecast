import streamlit as st
import pandas as pd
import requests
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

# Set Streamlit page config at the very beginning
st.set_page_config(page_title="PV Forecast", layout="wide")

# 1. Define Location (Rome, Italy)
latitude = 45.73263
longitude = 13.69278
tz = 'Europe/Rome'
location = Location(latitude, longitude, tz=tz)

# 2. Define System Specifications
temperature_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

system = PVSystem(
    surface_tilt=30,
    surface_azimuth=180,
    module_parameters={'pdc0': 6300, 'gamma_pdc': -0.004},
    inverter_parameters={'pdc0': 6000},
    temperature_model_parameters=temperature_parameters
)

mc = ModelChain(system, location, aoi_model='physical', spectral_model='no_loss')

# 3. Fetch Real Weather Data
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": latitude,
    "longitude": longitude,
    "current": "temperature_2m,wind_speed_10m,weather_code",
    "hourly": "shortwave_radiation,direct_normal_irradiance,diffuse_radiation,temperature_2m,wind_speed_10m",
    "timezone": tz,
    "forecast_days": 3
}

response = requests.get(url, params=params)
data = response.json()

# --- ERROR HANDLING ---
# Catch API errors (like rate limits or invalid parameters) before they crash the app
if "hourly" not in data:
    st.error("⚠️ The Weather API returned an error instead of data.")
    st.json(data)  # Display the exact API error message in the Streamlit app
    st.stop()  # Stop execution safely
# ----------------------

# 4. Format the API Data for pvlib
weather = pd.DataFrame({
    'ghi': data['hourly']['shortwave_radiation'],
    'dni': data['hourly']['direct_normal_irradiance'],
    'dhi': data['hourly']['diffuse_radiation'],
    'temp_air': data['hourly']['temperature_2m'],
    'wind_speed': data['hourly']['wind_speed_10m']
})

# Convert timestamps
weather.index = pd.to_datetime(data['hourly']['time'])
weather.index = weather.index.tz_localize(tz)

# 5. Run the Forecast Model
mc.run_model(weather)
forecasted_power = mc.results.ac.fillna(0)  # AC Power output in Watts

# 6. Extract Current Weather Conditions
current_data = data['current']
wmo_codes = {
    0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
    45: 'Fog', 48: 'Depositing rime fog', 51: 'Light drizzle', 53: 'Moderate drizzle',
    55: 'Dense drizzle', 56: 'Light freezing drizzle', 57: 'Dense freezing drizzle',
    61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain', 66: 'Light freezing rain',
    67: 'Heavy freezing rain', 71: 'Slight snow', 73: 'Moderate snow', 75: 'Heavy snow',
    77: 'Snow grains', 80: 'Slight rain showers', 81: 'Moderate rain showers',
    82: 'Violent rain showers', 85: 'Slight snow showers', 86: 'Heavy snow showers',
    95: 'Thunderstorm', 96: 'Thunderstorm with slight hail', 99: 'Thunderstorm with heavy hail'
}
current_temp = current_data['temperature_2m']
current_wind = current_data['wind_speed_10m']
current_sky = wmo_codes.get(current_data['weather_code'], "Unknown")

# 7. Group by Day and Visualize
grouped_power = forecasted_power.groupby(forecasted_power.index.date)
unique_days = list(grouped_power.groups.keys())

# Generate styled subplot titles
subplot_titles = []
for date in unique_days:
    daily_energy_kwh = grouped_power.get_group(date).sum() / 1000
    subplot_titles.append(
        f"📅 {date.strftime('%B %d, %Y')} &nbsp;&nbsp;|&nbsp;&nbsp; ⚡ Total: {daily_energy_kwh:.2f} kWh")

# Create a 3-row interactive figure WITH secondary Y-axes
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=subplot_titles,
    vertical_spacing=0.08,
    specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]]
)

for i, date in enumerate(unique_days):
    daily_data = grouped_power.get_group(date)

    # Calculate hourly and cumulative energy
    hourly_kwh = daily_data.values / 1000
    cumulative_kwh = pd.Series(hourly_kwh).cumsum().values

    # Trace 1: Power output in Watts (Primary Y-Axis - Filled Area)
    fig.add_trace(
        go.Scatter(
            x=daily_data.index,
            y=daily_data.values,
            mode='lines',
            name='Power (W)',
            line=dict(color='#FFD700', width=3, shape='spline'),
            fill='tozeroy',
            fillcolor='rgba(255, 215, 0, 0.15)',
            customdata=hourly_kwh,
            hovertemplate="<b style='color:#FFD700'>Power Output:</b> %{y:.0f} W<br><b>Energy (this hr):</b> %{customdata:.2f} kWh<extra></extra>",
            showlegend=(i == 0)
        ),
        row=i + 1, col=1, secondary_y=False
    )

    # Trace 2: Cumulative Energy in kWh (Secondary Y-Axis)
    fig.add_trace(
        go.Scatter(
            x=daily_data.index,
            y=cumulative_kwh,
            mode='lines',
            name='Cumulative Energy (kWh)',
            line=dict(color='#00E5FF', width=3, dash='dot', shape='spline'),
            hovertemplate="<b style='color:#00E5FF'>Total Accumulated:</b> %{y:.2f} kWh<extra></extra>",
            showlegend=(i == 0)
        ),
        row=i + 1, col=1, secondary_y=True
    )

    # Format axes for each subplot
    fig.update_xaxes(
        tickformat="%H:%M", row=i + 1, col=1,
        showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', zeroline=False
    )
    fig.update_yaxes(
        title_text="Power (Watts)", row=i + 1, col=1, secondary_y=False,
        showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', zeroline=False,
        title_font=dict(color='#FFD700')
    )
    fig.update_yaxes(
        title_text="Cumulative (kWh)", row=i + 1, col=1, secondary_y=True,
        showgrid=False, zeroline=False,
        title_font=dict(color='#00E5FF')
    )

# Apply global dark theme and overarching styling
fig.update_layout(
    template="plotly_dark",
    title=dict(
        text=f"<span style='font-size:26px; color:#FFD700;'><b>☀️ PV Production Forecast & Cumulative Output</b></span><br>"
             f"<span style='font-size:14px; color:#A0A0A0;'><i>📍 6.3 kW System &nbsp;|&nbsp; 🌡️ {current_temp}°C &nbsp;|&nbsp; 💨 {current_wind} km/h &nbsp;|&nbsp; ☁️ {current_sky}</i></span>",
        x=0.5,
        xanchor='center'
    ),
    height=1050,
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="rgba(20, 20, 20, 0.9)",
        bordercolor="#444",
        font_size=14,
        font_family="Segoe UI, Arial, sans-serif"
    ),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1,
        bgcolor="rgba(0,0,0,0)"
    ),
    margin=dict(t=130, b=40, l=40, r=40),
    font=dict(family="Segoe UI, Arial, sans-serif")
)

# 8. Calculate total energy BEFORE rendering in Streamlit
total_energy_kwh = forecasted_power.sum() / 1000

# 9. Render Streamlit Elements
st.plotly_chart(fig, use_container_width=True)
st.success(f"Total forecasted energy for the next 3 days: **{total_energy_kwh:.2f} kWh**")