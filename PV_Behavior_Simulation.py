# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:47:29 2024

@author: crhoades
"""

#******************************************************************************************************************************************************************************************************
"""
This program models the behavior of a PV system designed with predefined load parameters.
The irradiation captured by the system is derived from the nearest TMY data station, as defined by "fname".
This simulation plots output, load, batterey state of charge (BSOC).
The simulation calculates System Availability, Average BSOC, Lowest BSOC and date, and the energy use fraction.
"""
#******************************************************************************************************************************************************************************************************

import matplotlib.pyplot as plt
import numpy as np
import photovoltaic as pv
import time

from numpy import radians as rad

#*************************************** CONSTANTS SECTION ********************************************************************************************************************************************

Arrays = 31 # super simple parameter for PV array output
Bateries = 17
Wise_frac = 0.50

# J0 and Jsc and n values from Project 2 (for simplicity)
J0 = 1.3983790003885948e-13 # A/cm²
JL = 0.03895419148337684  # A/cm²
Area = 243.36 # cm²
Rseries = 0.6935010143333081 #ohm.cm
voc=pv.cell.Voc(JL,J0)
ff_0 = 0.8427615589553071
cells = 72 * Arrays

Pmax = 285 * Arrays
# Define Module tilt and module azimuth

module_azimuth = 197  # point the module south (180)
module_tilt = 30


#*************************************** LOCATION SECTION ********************************************************************************************************************************************

# Read TMY data file, import by station name
fname = '724275TYA.CSV'

station, GMT_offset, latitude, longitude, altitude = np.genfromtxt(fname, max_rows=1, delimiter=",", usecols=(0, 3, 4, 5, 6))
location_name, location_state = np.genfromtxt(fname, max_rows=1, delimiter=",", usecols=(1,2), dtype= str)
ETR, GHI, DNI, DHI, ambient_temperature = np.genfromtxt(fname, skip_header=2, delimiter=",", usecols=(2,4,7, 10,31), unpack=True)

city = location_name

module_elevation = latitude

print('station number: ', station)
print('City: ', city)
print('GMT: ', GMT_offset)
print('Latitude (degrees):', latitude)
print('Longitude (degrees):', longitude)
print('Altitude (m): ', altitude)

# details for the plots which makes string to label the plots
details4plot = ' lat: {:.1f}°, long: {:.1f}°, Module el: {}°, az: {}°'.format(
    latitude, longitude, module_elevation, module_azimuth)
print(details4plot)


#*************************************** SOLAR ANGLE SECTION ********************************************************************************************************************************************

# The big part of these calculations is getting the day and hour correct.

# Some ways to get the day and hours into an array  (1) read date and time from data file into an array that has day number and then hour of the day;
# :)                                                (2) Generate an array with day number and hour of the year. 

# The following generates two arrays, each with 8760 data points. The hours of the day array has a repeating sequence of hours
# We generate the hours 1 to 24 365 times. So the array  goes 1,2, ... 24, 1, 2, 3, ... 24,  Total 8760 data points.
hours = np.arange(1,25) # Generate an array with the values 1 to 24
hour_of_day = np.tile(hours,365) # Generate an array with the hours 1 to 24 repeated 365 times

# Create an array  day 1 is repeated 24 times, day2 is repeated 24 times etc, so have 1 24 times, then 2 24 times,etc.
days = np.arange(1,366)
day_no = np.repeat(days,24)

elevations, azimuths = pv.sun.sun_position(day_no, latitude, longitude, GMT_offset, hour_of_day, 0)
fraction_normal_to_module = pv.sun.module_direct(azimuths,elevations,module_azimuth,module_elevation)
DNI_module = DNI * fraction_normal_to_module
diffuse_module = DHI * (180 - module_elevation) / 180
total_module = diffuse_module + DNI_module


#*************************************** OUTPUT SECTION ********************************************************************************************************************************************

curr = JL * Area
max_power = curr * voc * cells * ff_0
#print("Cell Current: ",current)
#print("Voc: ",voc)
print("\nMaximum Power of 72 cells at AM1.5 : ",max_power,"\n")


# Calculate the light intensity;
# Since JL from project 2 has been calculated at AM1.5G (1000W/m²), 
# we need to divide by 1000 and multiply by the DHI and DNI to find JL for every hour of the year
# The unit of JL_module is A/cm²
JL_module = (DNI+DHI)*0.001*JL

# This prints the current in mA/cm² from the module for every hour of the year
if 1:  # 1 - plot the data, 0 - turn plot off
    plt.figure() 
    plt.title('Current Density from the Module    AM1.5G Jsc = {:.2f} mA/cm²'.format(1e3*JL))
    plt.plot(1e3*JL_module)
    plt.xlabel('hour of the year')
    plt.ylabel('Current Density in hour interval (mA/cm²)')
    plt.show()

# We need to multiply by 1e4 to convert from 1/cm² to 1/m² 
# and divide by 1e3 to convert from W to kW, resulting in a factor of 10
P_module = 10*JL_module*voc*ff_0

print('\ncapacity factor = ', 100*sum(P_module)/(10*8760*JL*voc*ff_0), '%')

#define pmodule pmax
P_module_Pmax = total_module*0.001*Pmax*10 / (Area*cells)

# This prints the power output in W from the module for every hour of the year
if 1:  # 1 - plot the data, 0 - turn plot off
    plt.figure()   
    plt.title('Annual Module Power, Ohio County @30 deg.tilt: {:.2f} kWh'.format(sum(P_module_Pmax)*Area*cells/1e4)+'(nom='+str(Pmax)+'W)')
    #plt.plot(total_module*0.001*Pmax/number_modules_series)
    plt.plot(P_module*Area/10)
    plt.xlabel('hour of the year')
    plt.ylabel('Power per module in hour interval (W)')
    plt.show()
    
# calculations for module amp hours with efficiency of converter considered
system_voltage = 48
MPPT_efficiency = 0.95
PV_Array_Ah = P_module*Area*cells/10 * MPPT_efficiency / system_voltage

# This prints the current output in Amps after conversion from the module for every hour of the year    
if 1:  # 1 - plot the data, 0 - turn plot off
    plt.figure()
    plt.title('Annual Module Current, Ohio County @30 deg.tilt: {:.2f} Ah'.format(sum(PV_Array_Ah))+'(nom='+str(Pmax)+'W)')
    plt.plot(PV_Array_Ah)
    plt.xlabel('hour of the year')
    plt.ylabel('Current from Modules in hour interval (A)')
    plt.show()
    
#*************************************** LOAD SECTION ********************************************************************************************************************************************
         
# ****************
# Calculate Load *
# ****************

# determine load #1 = % of 1A needed to run the system when all lights are on
bulb_wattage = 3 
shoplight_wattage = 24 
total_load_1_wattage = bulb_wattage*40 + shoplight_wattage*2
normalized_load_1 = total_load_1_wattage / system_voltage
    
  
print("Load 1: ",normalized_load_1)

    
# determine load #2 = % of 1A needed to run the system when all screens are on
screen_wattage = 100 
computer_wattage = 200 
phone_charging_wattage = 20
total_load_2_wattage = computer_wattage + screen_wattage + phone_charging_wattage*6
normalized_load_2 = total_load_2_wattage / system_voltage


print("Load 2: ",normalized_load_2)

# determine load #3 = % of 1A needed to run the system when Air Conditioner is on
AC_wattage = 3500
total_load_3_wattage = AC_wattage
normalized_load_3 = total_load_3_wattage / system_voltage


print("Load 3: ",normalized_load_3)


# determine load #4 = % of 1A needed to run the system when large appliances are on
Clotheswasher_wattage = 1000
Dryer_wattage = 3000
Dishwasher_wattage = 1800
Oven_wattage = 3500
total_load_4_wattage = Clotheswasher_wattage + Dryer_wattage + Dishwasher_wattage + Oven_wattage
normalized_load_4 = total_load_4_wattage / system_voltage

  
print("Load 4: ",normalized_load_4)


# determine load #5 = % of 1A needed to run the system when fridge/freezer is on
fridge_wattage = 200
freezer_wattage = 350 
total_load_5_wattage = freezer_wattage + fridge_wattage
normalized_load_5 = total_load_5_wattage / system_voltage

  
print("Load 5: ",normalized_load_5)


# determine load #6 = % of 1A needed to run the system when battery charger is on
Charger_wattage = 240
total_load_6_wattage = Charger_wattage * 2
normalized_load_6 = total_load_6_wattage / system_voltage


print("Load 6: ",normalized_load_6)


# determine load #7 = % of 1A needed to run the system when water pumps are on
Pump_wattage = 60
total_load_7_wattage = Pump_wattage * 2
normalized_load_7 = total_load_7_wattage / system_voltage


print("Load 7: ",normalized_load_7)



# initialize arrays


Load = np.zeros(len(PV_Array_Ah))       # Set up an array with zeros for Load in Ah

Load_wise_use = np.zeros(len(PV_Array_Ah)) # Set up an array with zeros for Load (wise use) in Ah

rise_plot = np.zeros_like(PV_Array_Ah)  # Set up an array with zeros for sunrise
set_plot = np.zeros_like(PV_Array_Ah)   # Set up an array with zeros for sunet

for idx, JLm in enumerate(PV_Array_Ah): # Make a for loop; idx is indexing parameter for hours of the year
    day_number = np.floor(idx/24)
    hour = idx - day_number*24
    # Use divmod for more simple code; use array shape above to avoid the for loops and re-calculating same thing 23 times

    # Find sunrise and sunsert and plot
    dec = pv.sun.declination(day_number)
    sunrise, sunset = pv.sun.sun_rise_set(latitude, dec, 0)  # 0 means use local solar time
    rise_plot[idx] = sunrise
    set_plot[idx] = sunset
    
#####################
# Define Load Times #
#####################

    ############# Normal Only Loads
    
    # Define load #1 (lights) as 'on' for 2 hrs before sunset until 10 pm
    if hour > sunset -2 and hour < 22:
        Load[idx] = normalized_load_1
        Load_wise_use[idx] = normalized_load_1
    else:
        Load[idx] = 0
        Load_wise_use[idx] = 0

    # Define load #2 (screens) as 'on' from 7 pm until 10 pm
    if hour > 19 and hour < 22:
        Load[idx] = Load[idx] + normalized_load_2
        Load_wise_use[idx] = Load_wise_use[idx] + normalized_load_2
    else:
        Load[idx] = Load[idx] + 0
        Load_wise_use[idx] = Load_wise_use[idx] + 0
       
    # Define load #3 (AC) as 'on' during daytime: 50% during warmer months and full power during summer
    if sunrise < hour < sunset and 150 < day_number < 250:
        Load[idx] = Load[idx] + normalized_load_3
        Load_wise_use[idx] = Load_wise_use[idx] + normalized_load_3
    elif  sunrise < hour < sunset and 100 < day_number < 300:
        Load[idx] = Load[idx] + (normalized_load_3*.5)
        Load_wise_use[idx] = Load_wise_use[idx] + (normalized_load_3*.5)
    else:
        Load[idx] = Load[idx] + 0
        Load_wise_use[idx] = Load_wise_use[idx] + 0
            
    # Define load #4 (Washer/Dryer/Oven/Dishwasher) as 'on' for one hour each at 7pm
    if hour ==19:
        Load[idx] = Load[idx] + normalized_load_4
        Load_wise_use[idx] = Load_wise_use[idx] + 0
    else:
        Load[idx] = Load[idx] + 0
        Load_wise_use[idx] = Load_wise_use[idx] + 0
        
    ############# Emergency And Normal Loads
        
    # Define load #5 (freezer) as 'on' for one hour and 'off' for 7 hours
    if hour % 8 == 0:
        Load[idx] = Load[idx] + normalized_load_5
        Load_wise_use[idx] = Load_wise_use[idx] + normalized_load_5
    else:
        Load[idx] = Load[idx] + 0
        Load_wise_use[idx] = Load_wise_use[idx] + 0
 
    ############## Emergency Only Loads
    """   
    # Define load #6 (Chargers) as 'on' for one hour at end of day year round
    if hour == 21:
        Load[idx] = Load[idx] + normalized_load_6
        Load_wise_use[idx] = Load_wise_use[idx] + normalized_load_6
    else:
        Load[idx] = Load[idx] + 0
        Load_wise_use[idx] = Load_wise_use[idx] + 0
        
    # Define load #7 (Pump) as 'on' 3 times a day for an hour
    if hour % 8 == 0:
        Load[idx] = Load[idx] + normalized_load_7
        Load_wise_use[idx] = Load_wise_use[idx] + normalized_load_7
    else:
        Load[idx] = Load[idx] + 0
        Load_wise_use[idx] = Load_wise_use[idx] + 0
    """  
        
        
# plot the load energy and calculate total load drawn over course of the year
if 1:  # 1 - plot the data, 0 - turn plot off
    plt.figure()
    plt.title('Annual Load on System: {:.2f} kWh'.format((sum(Load) * system_voltage)/1e3))
    plt.plot(Load * system_voltage)
    plt.xlabel('hour of the year')
    plt.ylabel('Power supplied to load in hour (W)')
    plt.show()
    

#*************************************** BATTERY SECTION *****************************************************************************************************************************************

# Battery Parameters
capacity = 100 * Bateries # battery capacity in Ah
DOD = 0.2 # Depth of discharge in fraction is the lowest the battery can go

# Initialize PV system state parameters
BSOC = np.zeros(len(PV_Array_Ah)+1)               # Battery state of charge (%) 
BSOC[0]= 1.0                                      # Start with battery fully charged
battery_to_load = np.zeros(len(PV_Array_Ah))      # Amp-hr from battery to load

# Calculate PV to Load
PV_to_load = np.minimum(PV_Array_Ah,Load)           # This implments that if PV>load, PV_to_load = Load, if PV < load  PV_to_load = PV


# Calculate BSOC and battery to load
for idx, PV_Ah in enumerate(PV_Array_Ah): # idx is indexing parameter for hours of the year, PV_Ah is Amp-hr from PV array 
    

    if BSOC[idx] < Wise_frac :
        # alter load to reflect wise use
        Load[idx] = Load_wise_use[idx]
        PV_to_load[idx] = np.minimum(PV_Array_Ah[idx],Load[idx])
        if PV_Ah >= Load_wise_use[idx]:
            # In this part of the if loop, there is extra Ah after supplying the load, so this goes TO the battery.
            BSOC[idx+1] = np.minimum(BSOC[idx]+ (PV_Ah - PV_to_load[idx])/capacity, 1.0)
        else:
            # Since PV < load, battery has to supply the remainder if there is enough capacity
            # if Load - PV can be drawn from the battery (i.e., this is smaller than remaining battery capacity), this is taken from battery
            # Otherwise, the remaining battery capacity is taken from battery   
            battery_to_load[idx] = np.minimum(Load_wise_use[idx]-PV_Ah,(BSOC[idx]-DOD)*capacity)
            BSOC[idx+1] = np.maximum(BSOC[idx] - ((Load_wise_use[idx] - PV_to_load[idx])/capacity), DOD)
            
    else:
        if PV_Ah >= Load[idx]:
            # In this part of the if loop, there is extra Ah after supplying the load, so this goes TO the battery.
            BSOC[idx+1] = np.minimum(BSOC[idx]+ (PV_Ah - PV_to_load[idx])/capacity, 1.0)
        else:
            # Since PV < load, battery has to supply the remainder if there is enough capacity
            # if Load - PV can be drawn from the battery (i.e., this is smaller than remaining battery capacity), this is taken from battery
            # Otherwise, the remaining battery capacity is taken from battery   
            battery_to_load[idx] = np.minimum(Load[idx]-PV_Ah,(BSOC[idx]-DOD)*capacity)
            BSOC[idx+1] = np.maximum(BSOC[idx] - ((Load[idx] - PV_to_load[idx])/capacity), DOD)

# Re-size BCOC to get rid of the one extra data point
BSOC=np.resize(BSOC,8760)

# plot the battery state of charge for each hour of the year
if 1:  # 1 - plot the data, 0 - turn plot off
    plt.figure()
    plt.title('Minimum BSOC over the course of the year: {:.2f}%'.format(np.min(BSOC*100)))
    plt.plot(BSOC*100)
    plt.xlabel('hour of the year')
    plt.ylabel('Battery State Of Charge (%)')
    plt.show()

PV_to_battery = np.minimum(PV_Array_Ah - PV_to_load, (1-BSOC)*capacity)
PV_unused = PV_Array_Ah - PV_to_load - PV_to_battery
load_unmet_2 = np.maximum((Load - PV_Array_Ah - battery_to_load),0)
availability = (sum(Load) - sum(load_unmet_2))/sum(Load)
fraction_used_solar = (sum(PV_Array_Ah)-sum(PV_unused))/sum(PV_Array_Ah)
fraction_unused_solar = (sum(PV_unused))/sum(PV_Array_Ah)

# get date list from TMY data
date = np.genfromtxt(fname, skip_header=2, delimiter=",", usecols=0, unpack=True, dtype=str)
# get and trim lowest BSOC date
date_lowest_BSOC = date[np.argmin(BSOC)]
date_lowest_BSOC = date_lowest_BSOC[:-5] # remove year

print('\nSystem Availability: {:.2f}%'.format(availability*100))
print('Average BSOC: {:.2f}%'.format(100*np.mean(BSOC)))
print('Lowest BSOC: {:.2f}%'.format(np.min(BSOC*100)),'On',date_lowest_BSOC)
print('Fraction of Energy Used/Generated: {:.2f}%'.format(fraction_used_solar*100))