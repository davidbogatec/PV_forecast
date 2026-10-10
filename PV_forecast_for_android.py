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

# 1. Set minimal, mobile-friendly config
st.set_page_config(page_title="PV Forecast", layout="centered")

# Define Location (Rome, Italy)
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

if "hourly" not in data:
    st.error("⚠️ Weather API error.")
    st.json(data)
    st.stop()

# 4. Format the API Data for pvlib
weather = pd.DataFrame({
    'ghi': data['hourly']['shortwave_radiation'],
    'dni': data['hourly']['direct_normal_irradiance'],
    'dhi': data['hourly']['diffuse_radiation'],
    'temp_air': data['hourly']['temperature_2m'],
    'wind_speed': data['hourly']['wind_speed_10m']
})

weather.index = pd.to_datetime(data['hourly']['time']).tz_localize(tz)

# 5. Run the Forecast Model
mc.run_model(weather)
forecasted_power = mc.results.ac.fillna(0)
total_energy_kwh = forecasted_power.sum() / 1000

# 6. Extract Current Weather Conditions
current_data = data['current']
wmo_codes = {
    0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
    61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain', 71: 'Slight snow',
    95: 'Thunderstorm'
}
current_temp = current_data['temperature_2m']
current_wind = current_data['wind_speed_10m']
current_sky = wmo_codes.get(current_data['weather_code'], "Unknown")

# --- MODERN STREAMLIT UI HEADER ---
st.markdown("<h2 style='text-align: center; margin-bottom: 0px;'>☀️ PV Forecast</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888; font-size: 14px; margin-top: 5px;'>📍 6.3 kW System</p>", unsafe_allow_html=True)

# Weather Metrics
cols = st.columns(3)
cols[0].metric("🌡️ Temp", f"{current_temp} °C")
cols[1].metric("💨 Wind", f"{current_wind} km/h")
cols[2].metric("☁️ Sky", current_sky)

# Total Yield Highlight
st.success(f"**⚡ 3-Day Expected Yield:** {total_energy_kwh:.1f} kWh")
st.markdown("---")
# ----------------------------------

# 7. Group by Day and Visualize
grouped_power = forecasted_power.groupby(forecasted_power.index.date)
unique_days = list(grouped_power.groups.keys())

subplot_titles = [
    f"{date.strftime('%b %d')} | ⚡ {grouped_power.get_group(date).sum() / 1000:.1f} kWh"
    for date in unique_days
]

fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=subplot_titles,
    vertical_spacing=0.1,
    specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]]
)

for i, date in enumerate(unique_days):
    daily_data = grouped_power.get_group(date)
    hourly_kwh = daily_data.values / 1000
    cumulative_kwh = pd.Series(hourly_kwh).cumsum().values

    # Trace 1: Power output (W)
    fig.add_trace(
        go.Scatter(
            x=daily_data.index,
            y=daily_data.values,
            mode='lines',
            name='Power (W)',
            line=dict(color='#FFD700', width=2, shape='spline'),
            fill='tozeroy',
            fillcolor='rgba(255, 215, 0, 0.15)',
            showlegend=(i == 0)
        ),
        row=i + 1, col=1, secondary_y=False
    )

    # Trace 2: Cumulative Energy (kWh)
    fig.add_trace(
        go.Scatter(
            x=daily_data.index,
            y=cumulative_kwh,
            mode='lines',
            name='Energy (kWh)',
            line=dict(color='#00E5FF', width=2, dash='dot', shape='spline'),
            showlegend=(i == 0)
        ),
        row=i + 1, col=1, secondary_y=True
    )

    # Lock zooming and clean axes
    fig.update_xaxes(
        tickformat="%H:%M", row=i + 1, col=1, fixedrange=True,
        showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', zeroline=False
    )
    fig.update_yaxes(
        row=i + 1, col=1, secondary_y=False, fixedrange=True,
        showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', zeroline=False,
        showticklabels=False
    )
    fig.update_yaxes(
        row=i + 1, col=1, secondary_y=True, fixedrange=True,
        showgrid=False, zeroline=False,
        showticklabels=False
    )

# 8. Mobile styling
fig.update_layout(
    template="plotly_dark",
    height=750, # Slightly reduced height since title moved out
    dragmode=False,
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        bgcolor="rgba(0,0,0,0)", font=dict(size=10)
    ),
    margin=dict(t=20, b=20, l=10, r=10), # Severely minimized top margin
    font=dict(family="Arial, sans-serif", size=10)
)

# 9. Render Streamlit Elements with Toolbar Disabled
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
