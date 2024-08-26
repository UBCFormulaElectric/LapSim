import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import csv
import datetime
import pickle
import scipy

############################################################################
# Mallory Moxham - UBC Formula Electric - July 2024
############################################################################

# FILE PATHS!
# For input constants, change to applicable name:
constantsInPath = "sim_inputs_and_outputs/LapSimConstants.csv"
# For output .csv - can change output file name at the end here :)
fullDataOutPath = "sim_inputs_and_outputs/dynamicsCalcs.csv"
# For output plots
outputPlotPath = "sim_inputs_and_outputs/"
# For simple summary outputs
summaryOutPath = "sim_inputs_and_outputs/"
# For input cell data
cellDataPath = "sim_inputs_and_outputs/cellSelectionData.csv"
# AMK data input
AMKData = "data_prep_scripts/AMK_data.json"
# Optimum Lap Data
OLData = "sim_inputs_and_outputs/OptimumLapSim61.40.csv"
# Optimum Lap Plot Path
OLPlotPath = "sim_inputs_and_outputs/"
# SoC Curve Path (only for P28A cause that's what we've committed to)
SoCPath = "data_prep_scripts/P28_SoC_curve.pkl"

# USER-INPUT CONSTANTS
# STRING CONSTANTS
string_constants = 4            # BASED ON THE NUMBER OF INPUTS THAT ARE STRINGS!
track_choice = None             # Chosen track model
regen_on = None                 # True/False - Regen on or off
cell_choice = None              # P30, P28A, P26A, P45
torque_speed = None             # Peak or Continuous

# CAR CONSTANTS
no_cells_car_mass = None        # kg - CAR MASS WITHOUT SEGMENTS INCLUDED
Af = None                       # m^2 - FRONTAL AREA
Al= None                        # m^2 - WING ELEMENT AREA FROM TOP DOWN
mu_rr = None                    # COEFFICIENT OF ROLLING RESISTANCE

# TRACTION CONSTANTS
traction_speed = None           # km/h - MAX SPEED AROUND RADIUS IN TRACTION TEST
traction_radius = None          # m - RADIUS OF TRACTION TEST
mu_longitudinal = None          # Coefficient of longitudinal friction
mu_lateral = None               # Coefficient of lateral friction
Cl = None                       # Lift coefficient
Cd = None                       # Draf coefficient

# BATTERY CONSTANTS
# DEPENDS ON STARTING CONDITIONS
initial_SoC = None              # % - INITIAL STATE OF CHARGE

# DEPENDS ON BATTERY CHOICE
n_converter = None              # converter efficiency
cell_water_area = None          # m^2 - WATER COOLING SURFACE OF CELL
cell_aux_factor = None          # kg/kWh - SEGMENT AUXILLARY MASS/ENERGY
max_power = None                # W - MAXIMUM 80 kW AS PER RULES

# !!! 
# THERMAL CONSTANTS
heatsink_air_area = None        # m^2 - AIR COOLING SURFACE OF CELL
heatsink_mass = None            # kg - TOTAL PACK HEATSINK MASS
heatsink_cv = None              # J/C*kg - HEATSINK MATERIAL SPECIFIC HEAT
air_temp = None                 # C - CONSTANT ASSUMED AIR TEMP
water_temp = None               # C - CONSTANT ASSUMED WATER TEMP
air_htc = None                  # W/C*m^2 - ASSUMED CONSTANT AIR HTC
water_htc = None                # W/C*m^2 - ASSUMED CONSTANT WATER HTC
thermal_resistance_SE = None    # K/W - ASSUMED SERIES ELEMENT THERMAL RESISTANCE
thermal_resistance_out = None   # K/W - EXTERNAL CELL THERMAL RESISTANCE
thermal_resistance_in = None    # K/W - INTERNAL CELL THERMAL RESISTANCE
cell_cv = None                  # J/kgK - CELL SPECIFIC HEAT CAPACITY

# !!!
# BUSSING CONSTANTS
bussing_length_unsplit = None   # m - LENGTH OF BUSSING BETWEEN BATTERY AND MOTOR (BEFORE VOLTAGE IS SPLIT)
bussing_length_split = None     # m - TOTAL LENGTH OF BUSSING AFTER HV HAS BEEN SPLIT
bussing_resistivity = None      # Ohm-m - RESISTIVITY OF BUSBAR
bussing_crossSecnArea = None    # m^2 - CROSS SECTIONAL AREA OF BUSBAR

######################################################################
# This allows an external user to have control over the constants without touching the code

# Open .csv and take constants as input:

# Open the file
with open(constantsInPath, 'r', newline='') as infile:
    reader = csv.reader(infile)
    dataList = list(reader)
    dataList.pop(0)             # Remove title row

    # convert to array
    dataArray = np.array(dataList)

    # Take out the valuable columns and convert to floats as necessary
    value_name = dataArray[string_constants:,0]
    value = dataArray[string_constants:,2]
    value = np.asarray(value, dtype = float)

    # Deal with the string input constants
    track_name = dataArray[0,0]
    track = dataArray[0,2]

    regen_name = dataArray[1,0]
    regen = dataArray[1,2]

    cell_name = dataArray[2,0]
    cell = dataArray[2,2]

    torqueSpeed_name = dataArray[3,0]
    torqueSpeed = dataArray[3,2]

# Now create variables for everything
for x, y in zip(value_name, value):
    globals()[x] = y

# Again, deal with the string input constants
globals()[track_name] = track
globals()[regen_name] = regen
globals()[cell_name] = cell
globals()[torqueSpeed_name] = torqueSpeed

###################################################################################
# Upload CELL DATA

# Open .csv
cellData = pd.read_csv(cellDataPath).set_index('Cell')

# Extract values for the section that you want!
cell_max_voltage = cellData.loc[cell_choice]['maxVoltage']          # V - SINGLE CELL MAX VOLTAGE
cell_nominal_voltage = cellData.loc[cell_choice]['nomVoltage']      # V - SINGLE CELL NOMINAL VOLTAGE
cell_min_voltage = cellData.loc[cell_choice]['minVoltage']          # V - SINGLE CELL MIN VOLTAGE
cell_max_current = cellData.loc[cell_choice]['maxCurrent']          # A - SINGLE CELL MAX CURRENT
cell_max_charge_current = cellData.loc[cell_choice]['maxChargeCurrent'] # A - SINGLE CELL MAX CHARGE CURRENT
max_capacity = cellData.loc[cell_choice]['capacity']                # Ah - SINGLE CELL MAX CAPACITY
max_CRate = cell_max_current / max_capacity                         # N/A - SINGLE CELL MAXIMUM DISCHARGE C-RATE
max_charge_CRate = cell_max_charge_current / max_capacity           # N/A - SINGLE CELL MAXIMUM CHARGE C-RATE
single_cell_ir = cellData.loc[cell_choice]['DCIR']                  # Ohms - SINGLE CELL DCIR
cell_mass = cellData.loc[cell_choice]['mass']                       # kg - SINGLE CELL MASS
num_parallel_cells = cellData.loc[cell_choice]['numParallel']       # N/A - NUMBER OF PARALLEL ELEMENTS
num_series_cells = cellData.loc[cell_choice]['numSeries']           # N/A - NUMBER OF SERIES ELEMENTS
expected_pack_mass = cellData.loc[cell_choice]['packWeight']        # kg - WEIGHT OF ACCUMULATOR

# Import SoC curve data
with open(SoCPath, "rb") as file:
    SoC_dict = pickle.load(file)

# Create column lists
SoC_currents = [-5.6, -2.8, 0.56, 2.8, 10, 20, 30]  # A
SoC_col_level0 = list(SoC_dict.keys())  # Key names for top level of dict
# For the second part - the first column is always the capacity and the second is the voltage

###################################################################################
# CALCULATED CONSTANTS

# CONSTANTS - SHOULD NOT NEED CHANGING
delta_d = 0.01                                                              # distance interval - m
g = 9.81                                                                    # m/s^2

# Motor
motor_choice = "AMK"                                                        # For now the sim is only function with AMK
GR = 14.33                                                                  # Gear Ratio - emrax
wheel_diameter = 18 * 0.0254                                                # m - wheel diameter
wheel_radius = wheel_diameter / 2                                           # m - wheel radius
rho_air = 1.204                                                             # air density: kg / m^3
v_air = 0                                                                   # air velocity: m/s
radsToRpm = 1 / (2 * math.pi) * 60                                          # rad/s --> rpm

# Battery Pack - Calculated Values
initial_SoC = initial_SoC / 100                                             # Convert to fractional form!!
cell_capacity_initial = max_capacity * initial_SoC                          # Ah (initial state of charge for single cell)
pack_capacity_initial = cell_capacity_initial * num_parallel_cells          # Ah (initial state of charge for pack)
num_cells = num_series_cells * num_parallel_cells                           # Total number of cells
pack_nominal_voltage = cell_nominal_voltage * num_series_cells              # V - Pack nominal voltage
pack_max_voltage = cell_max_voltage * num_series_cells                      # V - Pack maximum voltage
pack_min_voltage = cell_min_voltage * num_series_cells                      # V - Pack minimum voltage
total_pack_ir = single_cell_ir / num_parallel_cells * num_series_cells      # ohms - total IR
knownTotalEnergy = pack_capacity_initial * pack_max_voltage / 1000          # kWh - maximum pack energy
max_current = num_parallel_cells * max_CRate * max_capacity                 # A - Max current through pack
max_charge_current = num_parallel_cells * cell_max_charge_current           # A - Max charge current through pack
only_cells_mass = cell_mass * num_parallel_cells * num_series_cells         # kg - mass of only cells
# !!! Total known energy is approximately SoC * nominal voltage * max capacity

# !!! LV Power Use
LV_power = 400                                                              # W - LV Power Use

# !!!
# Bussing calculations
bus_R_unsplit = bussing_resistivity * bussing_length_unsplit / bussing_crossSecnArea    # Ohms
bus_R_split = bussing_resistivity * bussing_length_split / bussing_crossSecnArea        # Ohms
bus_R_total = bus_R_unsplit + bus_R_split / 2                                           # Ohms - presuming that we split HV into 2

# Car Mass - Calculated Values
cell_aux_mass = cell_aux_factor * pack_nominal_voltage * pack_capacity_initial / 100  # kg - at the moment, based on nominal energy
mass = no_cells_car_mass + expected_pack_mass                               # kg

# Thermals - Calculated Values
battery_heat_capacity = cell_cv*cell_mass                                   # J/C
air_tc = air_htc*heatsink_air_area                                          # W/C
water_tc = water_htc*cell_water_area                                        # W/C
air_thermal_resistance = 1 / air_tc                                         # K/W
heatsink_temp_0 = air_temp                                                  # C
batteryTemp0 = air_temp                                                     # C - starting temperature of battery pack (may change if necessary)

# Traction Constants
# at 30 km/h, we travelled around a 5 m radius circle
a_centrip = (traction_speed * 1000 / 3600)**2 / traction_radius             # v^2 / r (convert to m/s)
test_mass = 225                                                             # kg - car mass used in testing
F_friction = test_mass * a_centrip                                          # calculate the friction force
mu_lateral = F_friction / (test_mass * g)                                   # calculate the tire friction coefficient

# Some more arbitrary speed measurements
max_traction_force = mass * g * mu_longitudinal                             # N - max force in LONGITUDINAL DIRECTION
F_friction = mu_lateral * mass * g                                          # friction force based on the evaluated car mass.
brake_force = 4000                                                          # N - could be altered to get a better output curve
brake_decel = brake_force / mass                                            # m/s2 - based on specific car mass, top deceleration rate will change

# Accel vs endurance
if track_choice == "Acceleration":                                         # Change the number of laps for each race
    numLaps = 1
    TRACK = "Sim_Acceleration.csv"
elif track_choice == "Autocross":
    numLaps = 1
    TRACK = "Sim_Autocross.csv"
elif track_choice == "SkidPad":
    numLaps = 1
    TRACK = "Sim_SkidPad.csv"
elif track_choice == "Endurance":
    numLaps = 22
    TRACK = "Sim_Endurance.csv"
else:
    print("Incorrect track chosen. Please choose one of: Acceleration, Autocross, SkidPad, Endurance.")

#################################################################################################
# FUNCTIONS
#################################################################################################

### quad_formala
## Only because I don't trust np.roots after getting incorrect output from it - read up that this is an issue with the fxn.
## Also, I wanted to control the way that I get output from the function.
def quad_formula(a, b, c):
    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        root1 = 0
        root2 = 0
    else:
        root1 = (-b + discriminant**(1/2)) / (2*a)
        root2 = (-b - discriminant**(1/2)) / (2*a)

    return [root1, root2]

### findClosestMatch
## function for finding closest match (rather than searchsorted)
def findClosestMatch(vector, x):
    # vector = vector # simplifying whatever form we were given as the vector
    index = np.searchsorted(vector, x)

    # based on the return value of searchsorted. We need to check THAT and the value below (with one edge case)
    if index == len(vector):
        index = index - 1      
    elif index != 0:
        if abs(x - vector[index]) > abs(x - vector[index - 1]):
            index = index - 1
    
    return index

### nextTime
## Solve for next time:
def nextTime(dataDict, i):
    # constant velocity edge case
    if dataDict['a_tan0'][i] == 0:
        dt = delta_d / dataDict['v0'][i]
    # constant acceleration edge case
    else:
        # Solve for next time
        poly_coeffs = np.array([1/2 * dataDict['a_tan0'][i], dataDict['v0'][i], -delta_d]) # OK - it is definitely something wrong with this!
        dt_pair = quad_formula(poly_coeffs[0], poly_coeffs[1], poly_coeffs[2])
        dt = min(dt_pair)
        if dt < 0:
            dt = max(dt_pair)

    return dt

### round_nearest
## Round for floating point inaccuracies
def round_nearest(x, a):
    return round(x / a) * a

### fastestNextSpeed
## the initial calculations to determine a speed and distance
def fastestNextSpeed(dataDict, TorqueSpeed, i, longitudinal_traction_limits):
    # angular frequency of wheel: w_wh
    dataDict['w_wh'][i] = dataDict['v0'][i] / wheel_radius

    # angular frequency of motor: w_m - also convert from rad/s to rpm
    dataDict['w_m'][i] = dataDict['w_wh'][i] * GR * radsToRpm

    # solve for motor torque: T_m
    index = findClosestMatch(TorqueSpeed.loc[:,'Speed'].to_list(), dataDict['w_m'][i])
    dataDict['T_m'][i] = TorqueSpeed.iloc[index, 1]

    # DEBUG
    #print("INITIAL TORQUE: %.3f. Trial: %d" % (dataDict['T_m'][i], i))
    # dataDict['T_initial_debug'][i] = dataDict['T_m'][i]

    # axel torque: T_a
    dataDict['T_a'][i] = dataDict['T_m'][i] * GR

    # Traction force: F_trac
    dataDict['F_trac'][i] = dataDict['T_a'][i] / wheel_radius * 2  # Traction force from FOUR motors
    # dataDict['F_trac'][i] = dataDict['T_a'][i] / wheel_radius  # Traction force from FOUR motors

    # NOT -> F_trac[i] = T_a[i] / (2 * wheel_radius)
    # MULTIPLIED BY TWO FOR 4WD!!

    # Down force
    dataDict['F_down'][i] = 1/2 * rho_air * Cl * Al * dataDict['v0'][i]

    # Determine max traction force from wheels: F_max = mu * F_normal = mu * (F_down + car weight force) 
    max_traction_force = mu_longitudinal * (dataDict['F_down'][i] + mass * g)

    # Now determine if the car is traction limited - if so, reduce torque applied to wheels.
    if dataDict['F_trac'][i] > max_traction_force:
        dataDict['F_trac'][i] = max_traction_force

        dataDict['T_a'][i] = wheel_radius * dataDict['F_trac'][i] / 2

        dataDict['T_m'][i] = dataDict['T_a'][i] / GR

        # debug
        # dataDict['T_traction_debug'][i] = dataDict['T_m'][i]

        #print("Limited by TRACTION. Max Torque: %.3f" % dataDict['T_m'][i])
        
        longitudinal_traction_limits = longitudinal_traction_limits + 1

    # Drag force: F_drag
    dataDict['F_drag'][i] = (rho_air * Af * Cd * (dataDict['v0'][i] + v_air)**2) / 2

    # Rolling resistance: F_RR = mu * normal force
    # Only when car is moving:
    if dataDict['v0'][i] == 0:
        dataDict['F_RR'][i] = 0
    else:
        dataDict['F_RR'][i] = mu_rr * mass * g

    # Fnet (tangential)
    dataDict['F_net_tan'][i] = dataDict['F_trac'][i] - (dataDict['F_drag'][i] + dataDict['F_RR'][i])

    # Acceleration (tangential)
    dataDict['a_tan0'][i] = dataDict['F_net_tan'][i] / mass

    # Solve for next time
    dt = nextTime(dataDict, i)          # 0 = 1/2 * a * t^2 + v0 * t + (y0-y1)
    dataDict['t0'][i+1] = dataDict['t0'][i] + dt

    # then for v1
    dataDict['v0'][i+1] = dataDict['v0'][i] + dataDict['a_tan0'][i] * dt

    # then for distance
    dataDict['r0'][i+1] = dataDict['r0'][i] + delta_d

    return dataDict, longitudinal_traction_limits

### findMaxSpeed
## determining the max speed requirement
def findMaxSpeed(trackData, dataDict, i):
    trackLocation = np.searchsorted(trackData['Cumulative Length'].values, dataDict['r0'][i])

    v_max = trackData['MaxVelocity'][trackLocation]

    return v_max

### limit_max_speed
## Based on the new max speed, what are our new variables
def limit_max_speed(dataDict, v_max, i, max_speed_limits):
    # Check if this force is greater than the maximum friction force. If so, then the car cannot speed up.
    # Then, we need to stay at the previous speed or we need to brake to reach a slower speed
    # So part of this is going to be to determine what the max speed is around the corner and then compare that to our speed.

    # Reset velocity to the max velocity
    dataDict['v0'][i+1] = v_max

    # Back calculate to determine values
    # Now determine the required acceleration at this point.
    dataDict['a_tan0'][i] = (dataDict['v0'][i+1]**2 - dataDict['v0'][i]**2) / (2 * delta_d)
    # dataDict['a_tan0'][i] = (dataDict['v0'][i+1] - dataDict['v0'][i]) / dt

    # Now, what is the net force
    dataDict['F_net_tan'][i] = mass * dataDict['a_tan0'][i]

    # Now based on the net force, what is the traction force sent to the wheels
    dataDict['F_trac'][i] = dataDict['F_net_tan'][i] + (dataDict['F_drag'][i] + dataDict['F_RR'][i])

    # axel torque
    dataDict['T_a'][i] = dataDict['F_trac'][i] * wheel_radius / 2
    # divide by two factor to represent 4WD

    # motor torque
    dataDict['T_m'][i] = dataDict['T_a'][i] / GR

    #print('Limited by MAX SPEED. Torque: %.3f' % (dataDict['T_m'][i]))
    max_speed_limits = max_speed_limits + 1

    # w_m, w_wh, v0 stay the same

    # New time also changes:
    dt = nextTime(dataDict, i)
    dataDict['t0'][i+1] = dataDict['t0'][i] + dt

    # debug
    # dataDict['T_maxspeed_debug'][i] = dataDict['T_m'][i]

    return dataDict, max_speed_limits

### braking
## Function to incorporate and calculate for braking iteratively
def braking(brakeDict, dataDict, i, braking_limits):
    # Find location
    trackLocation = findClosestMatch(brakeDict['Distance'], dataDict['r0'][i])

    if trackLocation > len(brakeDict['Distance']) - 1:
        trackLocation = trackLocation - 1

    if dataDict['v0'][i+1] > brakeDict['Speed'][trackLocation]:
        # Reset the speed to the braking speed if necessary
        dataDict['v0'][i+1] = brakeDict['Speed'][trackLocation]

        # Back calculate additional parameters - TAKE NOTE LATER IF REGEN IS ON OR NOT
        # IF REGEN OFF, THEN ENSURE THAT NEGATIVE TORQUE --> NO BATTERY POWER
        # Now determine the required acceleration at this point.
        dataDict['a_tan0'][i] = (dataDict['v0'][i+1]**2 - dataDict['v0'][i]**2) / (2 * delta_d)
        # dataDict['a_tan0'][i] = (dataDict['v0'][i+1] - dataDict['v0'][i]) / dt

        # Now, what is the net force
        dataDict['F_net_tan'][i] = mass * dataDict['a_tan0'][i]

        # Now based on the net force, what is the traction force sent to the wheels
        dataDict['F_trac'][i] = dataDict['F_net_tan'][i] + (dataDict['F_drag'][i] + dataDict['F_RR'][i])

        # axel torque
        dataDict['T_a'][i] = dataDict['F_trac'][i] * wheel_radius / 2
        # divide by two factor to represent 4WD

        # motor torque (SHOULD BE NEGATIVE!!)
        dataDict['T_m'][i] = dataDict['T_a'][i] / GR

        # w_m, w_wh, v0 stay the same

        # New time also changes:
        dt = nextTime(dataDict, i)
        dataDict['t0'][i+1] = dataDict['t0'][i] + dt

        # debug
        #print('limited by BRAKING. New torque = %.3f' % dataDict['T_m'][i])
        braking_limits = braking_limits + 1

    return dataDict, braking_limits

### batteryPower
## NEW BATTERY FUNCTION
def batteryPower(dataDict, i, ShaftTorque, MotorPower, AMK_speeds, regen_current_limits, AMK_current):
    # Calculate max charge current based on capacity
    max_charge_current_now = max_charge_CRate * dataDict['Pack Capacity'][i]

    # separate into regen/non-regen
    if dataDict['T_m'][i] < 0:
        if regen_on == "TRUE":
            # Check if the torque is below (in the negative direction) the rules limit of -5 Nm
            if dataDict['T_m'][i] > -5:
                # just set the output to ZERO (with only LV Power Usage)
                dataDict['P_battery'][i] = LV_power
                dataDict['Pack Current'][i] = dataDict['P_battery'][i] / dataDict['Pack Voltage'][i]
                dataDict['Drooped Voltage'][i] = dataDict['Pack Voltage'][i] - dataDict['Pack Current'][i] * total_pack_ir
            else:   # include regen
                # Determine power output from motor
                RPM_index = findClosestMatch(AMK_speeds, dataDict['w_m'][i])
                Torque_index = findClosestMatch(ShaftTorque.iloc[RPM_index, :].to_list(), abs(dataDict['T_m'][i]))  # switching the torque to be positive for this calculation
                P_3phase = MotorPower.iloc[RPM_index, Torque_index]

                # Determine power output from inverter
                P_outofInverter = P_3phase * n_converter

                # !! KEEP THIS TEXT !!
                # <<< Battery Current Calculation >>>
                # P_battery = 4*Pinv - I^2 * Rbussing
                # ALSO: P_battery = I*Pack Voltage - I^2 * DCIR_cells
                # Set these two equal to determine current (I): 4*Pinv - I2*Rbus = I*Pack Voltage - I2*Rcells, solve quadratic

                # Determine current into pack: - total_pack_ir
                current_pair = quad_formula((bus_R_total), dataDict['Pack Voltage'][i], -4*P_outofInverter)
                dataDict['Pack Current'][i] = -current_pair[0]                                                    # Setting negative current

                # Check current limit
                if abs(dataDict['Pack Current'][i]) > max_charge_current_now:
                    regen_current_limits = regen_current_limits + 1             # increase the count of regen limits
                    dataDict['Pack Current'][i] = -max_charge_current_now       # negative to indicate charging

                    # Correct the negative motor torque - need to fix this
                    current_index = findClosestMatch(AMK_current, abs(dataDict['Pack Current'][i]))
                    dataDict['T_m'][i] = -ShaftTorque.iloc[RPM_index, current_index]
                    
                # Reset new battery input power
                dataDict['P_battery'][i] = -(dataDict['Pack Voltage'][i] * abs(dataDict['Pack Current'][i])) + LV_power
                # else:
                #     # Otherwise, set power
                #     # Switch to REMOVE instead of ADD the bussing voltage, also setting negative power to indicate regen
                #     dataDict["P_battery"][i] = -(4 * P_outofInverter - dataDict['Pack Current'][i]**2 * bus_R_total)
                
                # Include losses
                dataDict['Total Losses'][i] = dataDict['Pack Current'][i]**2 * (bus_R_total + total_pack_ir)
                dataDict['Drooped Voltage'][i] = dataDict['Pack Voltage'][i] - dataDict['Pack Current'][i] * total_pack_ir
        else: 
            # just set the output to ZERO (with only LV Power usage)
            dataDict['P_battery'][i] = LV_power
            dataDict['Pack Current'][i] = dataDict['P_battery'][i] / dataDict['Pack Voltage'][i]
            dataDict['Drooped Voltage'][i] = dataDict['Pack Voltage'][i] - dataDict['Pack Current'][i] * total_pack_ir
    else:
        # determine current pulled from each motor
        RPM_index = findClosestMatch(AMK_speeds, dataDict['w_m'][i])
        Torque_index = findClosestMatch(ShaftTorque.iloc[RPM_index, :].to_list(), dataDict['T_m'][i])
        P_3phase = MotorPower.iloc[RPM_index, Torque_index]

        # Power into the inverter
        P_intoInverter = P_3phase / n_converter

        # !! KEEP THIS TEXT !!
        # <<< Battery Current Calculation >>>
        # P_battery = 4*Pinv + I^2 * Rbussing
        # ALSO: P_battery = I * Pack Voltage = I * (pack_OC_voltage - I * DCIR_cells) = I*packOCVoltage - I^2*DCIR_cells
        # Set these two equal to determine current (I): 4*Pinv + I2*Rbus = I*packOCVoltage - I2*Rcells, solve quadratic
        # For now, I'll leave it at pack nominal voltage, until we get better SoC prediction - then I'll use OC voltage
                # Next step will be to transform plots from Molicel into LookUp Tables for SoC determination

        # + total_pack_ir
        current_pair = quad_formula((bus_R_total), -dataDict['Pack Voltage'][i], 4*P_intoInverter)
        dataDict['Pack Current'][i] = current_pair[1]   # CHANGE LATER TO DETERMINE WHICH IS WHICH!!
        dataDict['P_battery'][i] = dataDict['Pack Voltage'][i] * dataDict['Pack Current'][i] + LV_power # - dataDict['Pack Current'][i]**2 * total_pack_ir
        dataDict['Drooped Voltage'][i] = dataDict['Pack Voltage'][i] - dataDict['Pack Current'][i] * total_pack_ir
        dataDict['Total Losses'][i] = dataDict['Pack Current'][i]**2 * (bus_R_total + total_pack_ir)

    return dataDict, regen_current_limits

### batteryChecks
## Battery safety checks
def batteryChecks(dataDict, i, AMK_current, AMK_speeds, ShaftTorque, power_limits, TotalLosses, MotorPower, current_limits):
    # Calculate max current based on capacity
    max_current_now = max_CRate * dataDict['Pack Capacity'][i]

    # Check for over 80 kW
    if dataDict['P_battery'][i] > max_power:
        dataDict['P_battery'][i] = max_power

        dataDict['Pack Current'][i] = max_power / dataDict['Pack Voltage'][i]
        P_intoInverter = (dataDict['P_battery'][i] - dataDict['Pack Current'][i]**2 * bus_R_total - LV_power) / 4
        P_intoMotor = P_intoInverter * n_converter

        # Determine resulting max torque
        RPM_index = findClosestMatch(AMK_speeds, dataDict['w_m'][i])
        Power_index = findClosestMatch(MotorPower.iloc[RPM_index, :].to_list(), P_intoMotor)
        dataDict['T_m'][i] = ShaftTorque.iloc[RPM_index, Power_index]
        
        ###############################################################################################
        # Now the rest of the values
        # axel torque: T_a
        dataDict['T_a'][i] = dataDict['T_m'][i] * GR

        # Traction force: F_trac
        dataDict['F_trac'][i] = dataDict['T_a'][i] / wheel_radius * 2  # Traction force from FOUR motors
        # NOT -> F_trac[i] = T_a[i] / (2 * wheel_radius)
        # MULTIPLIED BY TWO FOR 4WD!!

        # Drag force: F_drag
        dataDict['F_drag'][i] = (rho_air * Af * Cd * (dataDict['v0'][i] + v_air)**2) / 2

        # Rolling resistance: F_RR = mu * normal force
        # Only when car is moving:
        if dataDict['v0'][i] == 0:
            dataDict['F_RR'][i] = 0
        else:
            dataDict['F_RR'][i] = mu_rr * mass * g

        # Fnet (tangential)
        dataDict['F_net_tan'][i] = dataDict['F_trac'][i] - (dataDict['F_drag'][i] + dataDict['F_RR'][i])

        # Acceleration (tangential)
        dataDict['a_tan0'][i] = dataDict['F_net_tan'][i] / mass

        # Solve for next time
        dt = nextTime(dataDict, i)
        dataDict['t0'][i+1] = dataDict['t0'][i] + dt

        # then for v1
        dataDict['v0'][i+1] = dataDict['v0'][i] + dataDict['a_tan0'][i] * dt

        # then for distance
        dataDict['r0'][i+1] = dataDict['r0'][i] + delta_d

        # DEBUG
        #print("Limited by POWER. Torque: %.3f" % (dataDict['T_m'][i]))
        power_limits = power_limits + 1
        # dataDict['T_batterylimit_debug'][i] = dataDict['T_m'][i]

        # Checking to see if voltage-drooping is present in the motor current measurement
        dataDict['Total Losses'][i] = dataDict['Pack Current'][i]**2 * (bus_R_total + total_pack_ir)
        dataDict['Drooped Voltage'][i] = dataDict['Pack Voltage'][i] - dataDict['Pack Current'][i] * total_pack_ir
        
        ##############################################################################################
    # check for over the current limit
    if dataDict['Pack Current'][i] >= max_current_now:
        dataDict['Pack Current'][i] = max_current_now
        
        # New battery power
        dataDict['P_battery'][i] = dataDict['Pack Voltage'][i] * dataDict['Pack Current'][i] # - dataDict['Pack Current'][i]**2 * total_pack_ir
        P_intoInverter = (dataDict['P_battery'][i] - dataDict['Pack Current'][i]**2 * bus_R_total - LV_power) / 4
        P_intoMotor = P_intoInverter * n_converter

        # Determine resulting max torque
        RPM_index = findClosestMatch(AMK_speeds, dataDict['w_m'][i])
        Power_index = findClosestMatch(MotorPower.iloc[RPM_index, :].to_list(), P_intoMotor)
        dataDict['T_m'][i] = ShaftTorque.iloc[RPM_index, Power_index]

        ###############################################################################################
        # Now the rest of the values
        # axel torque: T_a
        dataDict['T_a'][i] = dataDict['T_m'][i] * GR

        # Traction force: F_trac
        dataDict['F_trac'][i] = dataDict['T_a'][i] / wheel_radius * 2  # Traction force from FOUR motors
        # NOT -> F_trac[i] = T_a[i] / (2 * wheel_radius)
        # MULTIPLIED BY TWO FOR 4WD!

        # Drag force: F_drag
        dataDict['F_drag'][i] = (rho_air * Af * Cd * (dataDict['v0'][i] + v_air)**2) / 2

        # Rolling resistance: F_RR = mu * normal force
        # Only when car is moving:
        if dataDict['v0'][i] == 0:
            dataDict['F_RR'][i] = 0
        else:
            dataDict['F_RR'][i] = mu_rr * mass * g

        # Fnet (tangential)
        dataDict['F_net_tan'][i] = dataDict['F_trac'][i] - (dataDict['F_drag'][i] + dataDict['F_RR'][i])

        # Acceleration (tangential)
        dataDict['a_tan0'][i] = dataDict['F_net_tan'][i] / mass

        # Solve for next time
        dt = nextTime(dataDict, i)
        dataDict['t0'][i+1] = dataDict['t0'][i] + dt

        # then for v1
        dataDict['v0'][i+1] = dataDict['v0'][i] + dataDict['a_tan0'][i] * dt

        # then for distance
        dataDict['r0'][i+1] = dataDict['r0'][i] + delta_d

        # DEBUG
        #print("Limited by CURRENT. Torque: %.3f" % (dataDict['T_m'][i]))
        current_limits = current_limits + 1
        # dataDict['T_batterylimit_debug'][i] = dataDict['T_m'][i]

        # Checking to see if voltage-drooping is present in the motor current measurement
        dataDict['Total Losses'][i] = dataDict['Pack Current'][i]**2 * (bus_R_total + total_pack_ir)
        dataDict['Drooped Voltage'][i] = dataDict['Pack Voltage'][i] - dataDict['Pack Current'][i] * total_pack_ir

        ##############################################################################################

    return dataDict, power_limits, current_limits

### SoCLookup
## To find the voltage at the next capacity point using binary search and interpolation
def SoCLookup(dataDict, i, current_index):
    min_voltage_warning = False # reset min voltage warning

    # Isolate the correct capacity array
    capacity_array = SoC_dict[SoC_col_level0[current_index]][:,0]   # Ah
    voltage_array = SoC_dict[SoC_col_level0[current_index]][:,1]    # V

    # Value to interpolate in this given set of values
    next_cell_capacity = dataDict['Pack Capacity'][i+1] / num_parallel_cells

    # To fix errors with going below minimum capacity:
    min_cell_capacity = 0.3
    if next_cell_capacity < min_cell_capacity:
        dataDict['Pack Capacity'][i+1] = min_cell_capacity * num_parallel_cells
        next_cell_capacity = min_cell_capacity

    f = scipy.interpolate.interp1d(capacity_array, voltage_array)   # create interpolation function
    next_cell_voltage = f(next_cell_capacity)                       # determine new

    dataDict['Pack Voltage'][i+1] = next_cell_voltage * num_series_cells

    # Just keeping the program running if it hits < 0% Soc
    if next_cell_voltage < cell_min_voltage:
        min_voltage_warning = True

    return dataDict, min_voltage_warning

### extraBatteryCalcs
## Battery thermals
def extraBatteryCalcs(dataDict, i):
    #####################
    # CAPACITY CALCULATIONS - Coulomb counting
    dt = (dataDict['t0'][i+1] - dataDict['t0'][i]) / 3600        # convert to units of hours
    dataDict['Pack Capacity'][i+1] = dataDict['Pack Capacity'][i] - dt * dataDict['Pack Current'][i]  # Determine next capacity based on capacity loss

    # Determine closest current value (no longer needs the absolute value because I've included charging curves)
    current_index = findClosestMatch(SoC_currents, dataDict['Pack Current'][i] / num_parallel_cells)

    # Determine next voltage
    dataDict, min_voltage_warning = SoCLookup(dataDict, i, current_index)

    # Calculate SoC
    dataDict['SoC Capacity'][i+1] = dataDict['Pack Capacity'][i+1] / (max_capacity * num_parallel_cells) * 100

    # Account for edge cases:
        # if voltage is below minimum voltage - just leave it at minimum voltage for now
        # if capacity is below minimum capacity - just leave it at that for now too!
    if dataDict['Pack Voltage'][i+1] < pack_min_voltage:
        dataDict['Pack Voltage'][i+1] = pack_min_voltage

    ########################################################################
    # Simple thermal calculations
    # P = I^2 * r - absolute of pack current to accout for regen also increasing pack temp
    dataDict['Cell Qgen'][i] = (abs(dataDict['Pack Current'][i]) / num_parallel_cells)**2 * single_cell_ir
    
    # NOTE TO SELF:
    # Look at my notes on confluence (https://ubcformulaelectric.atlassian.net/wiki/spaces/UFE/pages/edit-v2/405438472)
    # for how I came across this formula (and to verify that it is correct)
    air_temp_K = air_temp + 273.15

    battery_temp_K = (((dataDict['Cell Qgen'][i] * thermal_resistance_out + air_temp_K) * dt
                    + (dataDict['Battery Temp'][i] + 273.15) * battery_heat_capacity * (thermal_resistance_in + thermal_resistance_out))
                    / (battery_heat_capacity * (thermal_resistance_in + thermal_resistance_out) + dt))
    
    dataDict['Battery Temp'][i+1] = battery_temp_K - 273.15
    #######################################################################
    return dataDict

### energyConsumed
## Energy consumed
def energyConsumed(dataDict, i):

    # Add up energy over time to get an approximation
    # Trapezoidal Approximation (to be more accurate)
    if i != 0:
        # Start with energy used
        # (a+b)/2 * dt
        thisEnergy = 1/2*(dataDict['P_battery'][i] + dataDict['P_battery'][i-1]) * (dataDict['t0'][i] - dataDict['t0'][i-1])
        thisEnergy_kWh = thisEnergy / 3600000   # convert to kWh
        dataDict['Energy Use'][i+1] = dataDict['Energy Use'][i] + thisEnergy_kWh

        # Add up energy losses too
        thisLoss = 1/2*(dataDict['Total Losses'][i] + dataDict['Total Losses'][i-1]) * (dataDict['t0'][i] - dataDict['t0'][i-1])
        thisLoss_kWh = thisLoss / 3600000   # convert to kWh
        dataDict['Total Losses NRG'][i+1] = dataDict['Total Losses NRG'][i] + thisLoss_kWh
    
    return dataDict

### plotDetails
## Plot details to make my life cleaner :))
def plotDetails(x_axis, y_axis, plotTitle, ax):
    ax.set_title(plotTitle)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.grid(True)

    return

### plotData
## Plot the data
def plotData(dataDict, currentTime):
    # Plot the data in this function
    ROWS = 2
    COLS = 2
    FIGWIDTH = 20
    FIGHEIGHT = 12

    # create subplots
    fig, ax = plt.subplots(ROWS, COLS, figsize=(FIGWIDTH, FIGHEIGHT))
    supTitle = "Point Mass Vehicle Simulation - " + track_choice + ".csv"
    fig.suptitle(supTitle)

    if track_choice == "Endurance":
        # Plot 1)
        # Battery Voltage vs Distance
        row = 0; col = 0
        x_axis = "Distance (m)"
        y_axis = "Pack Voltage (V)"
        plotTitle = "Pack Voltage vs Distance"
        ax[row][col].plot(dataDict["r0"], dataDict["Pack Voltage"], label="Pack Voltage")       # plot the data
        # Will also plot a red line to show the minimum voltage
        ax[row][col].plot(dataDict['r0'], np.ones_like(dataDict['r0']) * pack_min_voltage, 'r')
        ax[row][col].legend()
        plotDetails(x_axis, y_axis, plotTitle, ax[row][col])  # add the details

        # Plot 4)
        # Battery Current vs time
        row = 0; col = 1
        x_axis = "Distance (m)"
        y_axis = "Battery Current (A)"
        plotTitle = "Battery Current vs Distance"
        ax[row][col].plot(dataDict["r0"], dataDict["Pack Current"])
        plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

        # Plot 3)
        # Battery power vs time
        row = 1; col = 0
        x_axis = "Distance (m)"
        y_axis = "Battery Power (kW)"
        plotTitle = "Battery Power vs Distance"
        ax[row][col].plot(dataDict["r0"], dataDict["P_battery"])
        plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

        # # Plot 4)
        # # Battery Drooped Voltage
        # row = 1; col = 1
        # x_axis = "Distance (m)"
        # y_axis = "SoC (%)"
        # plotTitle = "SoC vs Distance"
        # ax[row][col].plot(dataDict['r0'], dataDict['SoC Capacity'])
        # plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

        # Plot 4)
        row = 1
        col = 1
        x_axis = "Distance (m)"
        y_axis = "Battery Temperature (C)"
        plotTitle = "Battery Temperature vs Distance"
        ax[row][col].plot(dataDict['r0'], dataDict['Battery Temp'])
        plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

    elif track_choice == "Autocross":
        # Plot 4)
        # Battery Current vs time
        row = 0; col = 0
        x_axis = "Distance (m)"
        y_axis = "Battery Current (A)"
        plotTitle = "Battery Current vs Distance"
        ax[row][col].plot(dataDict["r0"], dataDict["Pack Current"])
        plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

        # Plot 2)
        # Velocity vs Distance
        row = 0; col = 1
        x_axis = "Distance (s)"
        y_axis = "Motor Torque (Nm)"
        plotTitle = "Torque vs Distance"
        ax[row][col].plot(dataDict["r0"], dataDict["T_m"])
        plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

        # Plot 3)
        # Battery power vs time
        row = 1; col = 0
        x_axis = "Distance (m)"
        y_axis = "Battery Power (kW)"
        plotTitle = "Battery Power vs Distance"
        ax[row][col].plot(dataDict["r0"], dataDict["P_battery"])
        plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

        # Plot 4)
        # Battery Drooped Voltage
        row = 1; col = 1
        x_axis = "Distance (m)"
        y_axis = "Drooped Voltage (V)"
        plotTitle = "Drooped Voltage vs Distance"
        ax[row][col].plot(dataDict['r0'], dataDict['Drooped Voltage'])
        # Will also plot a red line to show the minimum voltage
        ax[row][col].plot(dataDict['r0'], np.ones_like(dataDict['r0']) * pack_min_voltage, 'r')
        plotDetails(x_axis, y_axis, plotTitle, ax[row][col])
    
    else:
        # Plot 4)
        # Battery Current vs time
        row = 0; col = 0
        x_axis = "Distance (m)"
        y_axis = "Battery Current (A)"
        plotTitle = "Battery Current vs Distance"
        ax[row][col].plot(dataDict["r0"], dataDict["Pack Current"])
        plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

        # Plot 2)
        # Velocity vs Distance
        row = 0; col = 1
        x_axis = "Distance (s)"
        y_axis = "Speed (km/h)"
        plotTitle = "Speed vs Distance"
        ax[row][col].plot(dataDict["r0"], dataDict["v0"])
        plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

        # Plot 3)
        # Battery power vs time
        row = 1; col = 0
        x_axis = "Distance (m)"
        y_axis = "Battery Power (kW)"
        plotTitle = "Battery Power vs Distance"
        ax[row][col].plot(dataDict["r0"], dataDict["P_battery"])
        plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

        # Plot 4)
        # Battery Drooped Voltage
        row = 1; col = 1
        x_axis = "Distance (m)"
        y_axis = "Torque (Nm)"
        plotTitle = "Motor Torque vs Distance"
        ax[row][col].plot(dataDict['r0'], dataDict['T_m'])
        plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

    return fig


############################################################################
# PREVIOUS CODE:

# OLD CELL THERMAL CALCULATIONS:
    # # Simple thermal calculations
    # # P = I^2 * r - absolute of pack current to accout for regen also increasing pack temp
    # dataDict['Cell Qgen'][i] = (abs(dataDict['Pack Current'][i]) / num_parallel_cells)**2 * single_cell_ir
    
    # # Previous power out of cell calculation
    # # cell_p_out = (air_tc)*(dataDict['Battery Temp'][i]-air_temp) + water_tc*(dataDict['Battery Temp'][i]-water_temp)

    # # Calculate time interval
    # dt = dataDict['t0'][i+1] - dataDict['t0'][i]

    # # Power out of cell to heatsink
    # cell_p_out = (dataDict['Battery Temp'][i] - dataDict['Heatsink Temp'][i]) / (thermal_resistance_SE * num_parallel_cells)

    # # First temp calculations
    # dataDict['Battery Temp'][i+1] = dataDict['Battery Temp'][i] + dt * (dataDict['Cell Qgen'][i] - cell_p_out) / (battery_heat_capacity)

    # # Heatsink temp calculations
    # # Split up calculation:
    # calculation1 = (dataDict['Battery Temp'][i] - dataDict['Heatsink Temp'][i]) * num_series_cells / thermal_resistance_SE - air_tc * (dataDict['Heatsink Temp'][i] - air_temp)
    # dataDict['Heatsink Temp'][i+1] = dataDict['Heatsink Temp'][i] + dt / (heatsink_mass * heatsink_cv) * calculation1