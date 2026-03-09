#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 02:20:01 2023

@author: mjp38
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants and Initial Conditions
evap_option = 'aerodynamic'  # Options: 'aerodynamic' or 'GardnerHillel'
precipitation_option = 'poisson'  # Options: 'constant' or 'poisson'
P_const = 1  # Constant Precipitation (mm)
lambda_ = 0.2  # Poisson process frequency
alpha_R = 5  # Average rainfall depth (mm)

# Other constants and initial conditions
d = 1e-6 * 1000 * 1000 / (np.pi ** 2 / 4) * 86400  # m2/s to mm/day hydraulic diffusivity
R = 0  # Runoff (mm) (mm)
beta = 1  # Leakage coefficient
b = 0.5  # Variable for leakage calculation
alpha = 0.2  # Wind speed coefficient
wind_speed = 5  # Wind speed (m/s)
porosity = 0.4  # Soil porosity
field_capacity = 0.3  # Field capacity
wilting_point = 0.1  # Wilting point
initial_moisture = 0.2 * porosity  # Initial normalized soil moisture (converted to volumetric)
timestep = 1  # Time step (day) (day)
time_period = 1000  # Time period for simulation (days) (days)
time = np.arange(1, time_period + 1, timestep)
Zr = 300  # Root zone depth (mm)

# For evapb calculations
rho3 = 1.2  # Assumed air density
ch = 0.001  # Assumed transfer coefficient
vs = wind_speed  # Assumed wind speed at reference height
qb = 20  # Assumed specific humidity at reference height
qs = 15  # Assumed specific humidity at surface
v_qprime = 0  # Assumed vertical gradient of specific humidity

# Initialize soil moisture and precipitation
soil_moisture = np.zeros(time_period)
soil_moisture[0] = initial_moisture
precipitation = np.zeros(time_period)
evaporation = np.zeros(time_period)
leakage = np.zeros(time_period)

# Precipitation calculation
if precipitation_option == 'constant':
    precipitation[:] = P_const
elif precipitation_option == 'poisson':
    for t in range(time_period):
        num_events = np.random.poisson(lambda_)
        rainfall_depths = np.random.exponential(alpha_R, num_events)
        precipitation[t] = np.sum(rainfall_depths)

# Soil Water Balance Calculation
for t in range(1, time_period):
    S = soil_moisture[t - 1]

    if evap_option == 'aerodynamic':
        
        epb = rho3 * ch * (vs * (qb - qs) - v_qprime)
        # evap_max_dry calculation (assuming single layer for simplicity)
        pot_evap_can = 5  # Assumed potential evaporation
        betadl = 0.5  # Assumed betadl value
        betad = 0.5  # Assumed betad value
        dz = Zr  # Assuming single layer with depth = Zr
        thetm = 0.25  # Assumed average soil moisture content
        dt = timestep
        # Inside the loop, where evap_option == 'aerodynamic':
        theta = soil_moisture[t-1]  # current volumetric soil moisture at time t
        thetm = wilting_point * porosity  # wilting  converted to volumetric soilmoisture
        pr = precipitation[t]  # precipitation at current time step

        # Calculate evap_max for the single layer
        evap_max = (theta - Zr * thetm) / timestep

    # Calculate evap_max_dry
        evap_max_dry = min(evap_max, 2.467 * d * (theta - thetm) / dz + pr)
        E = epb
        #E = alpha * wind_speed  # (mm/day)
    elif evap_option == 'GardnerHillel':
            E = (np.pi**2 / 4) * d * S

    
    # Calculate leakage using the power law relationship
    c = 2 * b + 3
    L = beta * (S - wilting_point) ** (c - 1)

    # Update soil moisture
    P = precipitation[t]
    dS_dt = (P - R - L - E) / Zr
    soil_moisture[t] = soil_moisture[t - 1] + dS_dt * timestep
    evaporation[t] = E
    leakage[t] = L

    # Check for saturation and field capacity
    if soil_moisture[t] > porosity:
        soil_moisture[t] = porosity
    elif soil_moisture[t] < wilting_point:
        soil_moisture[t] = wilting_point

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Soil Moisture
axs[0, 0].plot(time, soil_moisture, 'b-', linewidth=2)
axs[0, 0].set_xlabel('Time (days)')
axs[0, 0].set_ylabel('Volumetric Soil Moisture')
axs[0, 0].set_title('Soil Moisture')

# Panel 2: Precipitation
axs[0, 1].plot(time, precipitation, 'g-', linewidth=2)
axs[0, 1].set_xlabel('Time (days)')
axs[0, 1].set_ylabel('Precipitation (mm)')
axs[0, 1].set_title('Precipitation')

# Panel 3: Evaporation
axs[1, 0].plot(time, evaporation, 'r-', linewidth=2)
axs[1, 0].set_xlabel('Time (days)')
axs[1, 0].set_ylabel('Evaporation (mm)')
axs[1, 0].set_title('Evaporation')

# Panel 4: Leakage
axs[1, 1].plot(time, leakage, 'm-', linewidth=2)
axs[1, 1].set_xlabel('Time (days)')
axs[1, 1].set_ylabel('Leakage (mm)')
axs[1, 1].set_title('Leakage')

plt.tight_layout()
plt.show()
