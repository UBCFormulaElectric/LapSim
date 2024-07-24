import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import csv
import datetime

############################################################################
# Mallory Moxham - UBC Formula Electric - July 2024
############################################################################

# FILE PATHS! (CHANGE THESE TO FIT YOUR COMPUTER PLEASE!)
# For input constants, change to applicable name:
constantsInPath = r'C:\VSCode_Python\LapSim_V2\sim_inputs_and_outputs\LapSimConstants.csv'
# For output .csv
fullDataOutPath = r'C:\VSCode_Python\LapSim_V2\sim_inputs_and_outputs\dynamicsCalcs.csv'
# For output plots
outputPlotPath = "C:\\VSCode_Python\\LapSim_V2\\sim_inputs_and_outputs\\"
# For simple summary outputs
summaryOutPath = "C:\\VSCode_Python\\LapSim_V2\\sim_inputs_and_outputs\\"
# For input cell data
cellDataPath = r'C:\VSCode_Python\LapSim_V2\sim_inputs_and_outputs\cellSelectionData.csv'
# AMK data input
AMKData = r'C:\VSCode_Python\LapSim_V2\data_prep_scripts\AMK_data.json'
# Optimum Lap Data
OLData = r'C:\VSCode_Python\LapSim_V2\sim_inputs_and_outputs\OptimumLapSim61.40.csv'
# Optimum Lap Plot Path
OLPlotPath = "C:\\VSCode_Python\\LapSim_V2\\sim_inputs_and_outputs\\"

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
air_factor_m = None             # kg/kg AIR COOLING MASS PER BATTERY MASS
water_factor_m = None           # kg/kg WATER COOLING MASS PER BATTERY MASS

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
max_capacity = cellData.loc[cell_choice]['capacity']                # Ah - SINGLE CELL MAX CAPACITY
max_CRate = cell_max_current / max_capacity                         # N/A - SINGLE CELL MAXIMUM DISCHARGE C-RATE
single_cell_ir = cellData.loc[cell_choice]['DCIR']                  # Ohms - SINGLE CELL DCIR
cell_mass = cellData.loc[cell_choice]['mass']                       # kg - SINGLE CELL MASS
battery_cv = cellData.loc[cell_choice]['batteryCv']                 # J/kgC - SINGLE CELL SPECIFIC HEAT CAPACITY
num_parallel_cells = cellData.loc[cell_choice]['numParallel']       # N/A - NUMBER OF PARALLEL ELEMENTS
num_series_cells = cellData.loc[cell_choice]['numSeries']           # N/A - NUMBER OF SERIES ELEMENTS

###################################################################################
# CALCULATED CONSTANTS

# CONSTANTS - SHOULD NOT NEED CHANGING
delta_d = 0.01                                                              # distance interval - m
g = 9.81                                                                    # m/s^2

# Motor
motor_choice = "AMK"                                                        # For now the sim is only function with AMK
if motor_choice == 'AMK':
    GR = 14.33                                                              # Gear Ratio - AMK
else:
    GR = 4.2                                                                # Gear Ratio - emrax
wheel_diameter = 18 * 0.0254                                                # m - wheel diameter
wheel_radius = wheel_diameter / 2                                           # m - wheel radius
rho_air = 1.204                                                             # air density: kg / m^3
v_air = 0                                                                   # air velocity: m/s
radsToRpm = 1 / (2 * math.pi) * 60                                          # rad/s --> rpm

# Battery Pack - Calculated Values
capacity0 = max_capacity * initial_SoC                                      # Ah (initial state of charge)
num_cells = num_series_cells * num_parallel_cells                           # Total number of cells
pack_nominal_voltage = cell_nominal_voltage * num_series_cells              # V - Pack nominal voltage
pack_max_voltage = cell_max_voltage * num_series_cells                      # V - Pack maximum voltage
pack_min_voltage = cell_min_voltage * num_series_cells                      # V - Pack minimum voltage
total_pack_ir = single_cell_ir / num_parallel_cells * num_series_cells      # ohms - total IR
knownTotalEnergy = initial_SoC * capacity0 * pack_nominal_voltage / 1000    # kWh - based on nominal voltage
max_current = num_parallel_cells * max_CRate * max_capacity                 # A - Max current through pack
# !!! Total known energy is approximately SoC * nominal voltage * max capacity

# !!!
# Car Mass - Calculated Values
total_cell_mass = cell_mass*num_cells                                       # kg
cooled_cell_mass = total_cell_mass*(1 + air_factor_m + water_factor_m)      # kg
cell_aux_mass = cell_aux_factor*(capacity0 * pack_nominal_voltage / 1000)   # kg
mass = no_cells_car_mass + total_cell_mass                                  # kg
#+ cooled_cell_mass + cell_aux_mass + heatsink_mass # kg

# !!! 
# Thermals - Calculated Values
battery_heat_capacity = battery_cv*cell_mass                                # J/C
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
    TRACK = "Sim_Autocross.csv"
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
    dataDict['T_initial_debug'][i] = dataDict['T_m'][i]

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
        dataDict['T_traction_debug'][i] = dataDict['T_m'][i]

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
    dataDict['T_maxspeed_debug'][i] = dataDict['T_m'][i]

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
def batteryPower(dataDict, i, ShaftTorque, MotorPower, TotalLosses, AMK_current, AMK_speeds):
    # separate into regen/non-regen
    if dataDict['T_m'][i] < 0:
        if regen_on == "TRUE":
            pass
        else: 
            # just set the output to ZERO
            dataDict['Pack Current'][i] = 0
            dataDict['P_battery'][i] = 0
    else:
        # determine current pulled from each motor
        RPM_index = findClosestMatch(AMK_speeds, dataDict['w_m'][i])
        Torque_index = findClosestMatch(ShaftTorque.iloc[RPM_index, :].to_list(), dataDict['T_m'][i])
        # single_motor_current = AMK_current[Torque_index]
        P_3phase = MotorPower.iloc[RPM_index, Torque_index]

        # Power into the inverter = P_motor-to-invert / n_converter
        P_intoInverter = P_3phase / n_converter

        # Then losses along the bussing line
        # Power from battery = Power into inverter
        # Taking currently known voltage, determine current running through pack
        # Now I'm a bit stuck on how to calculate losses that involve current: IR losses and bussing losses - but they could be done separately
        # The question is... since we assume power stays constant through the inverter - how do we calculate current into the inverter? Anyway...
        # Cause that would allow us to calculate voltage-drooping
        
        dataDict['P_battery'][i] = P_intoInverter * 4   # right now without losses
        dataDict['Pack Current'][i] = dataDict['P_battery'][i] / pack_nominal_voltage

    return dataDict

### batteryChecks
## Battery safety checks
def batteryChecks(dataDict, i, AMK_current, AMK_speeds, ShaftTorque, power_limits, TotalLosses, MotorPower, current_limits):
    # Compare power limit with overcurrent fault:
    max_current_power_limited = max_power / pack_nominal_voltage

    # if the power limit is more conservative
    if max_current_power_limited < max_current:
        # Check for over 80 kW
        if dataDict['P_battery'][i] > max_power:
            dataDict['P_battery'][i] = max_power

            dataDict['Pack Current'][i] = dataDict['P_battery'][i] / pack_nominal_voltage
            P_intoInverter = dataDict['P_battery'][i] / 4
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
            dataDict['T_batterylimit_debug'][i] = dataDict['T_m'][i]

            # Checking to see if voltage-drooping is present in the motor current measurement
            if dataDict['Pack Current'][i] != 0:
                dataDict['Drooped Voltage'][i] = dataDict['P_battery'][i] / dataDict['Pack Current'][i]
            
            ##############################################################################################
    
    # If the current limit is more conservative
    else:
        # check for over the current limit
        if dataDict['Pack Current'][i] >= max_current:
            dataDict['Pack Current'][i] = max_current
            
            # New battery power
            dataDict['P_battery'][i] = pack_nominal_voltage * dataDict['Pack Current'][i]
            P_intoInverter = dataDict['P_battery'][i] / 4
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
            dataDict['T_batterylimit_debug'][i] = dataDict['T_m'][i]

            # Checking to see if voltage-drooping is present in the motor current measurement
            if dataDict['Pack Current'][i] != 0:
                dataDict['Drooped Voltage'][i] = dataDict['P_battery'][i] / dataDict['Pack Current'][i]

            ##############################################################################################

    return dataDict, power_limits, current_limits

### energyConsumed
## Energy consumed
def energyConsumed(dataDict, i):

    # Add up energy over time to get an approximation
    # Trapezoidal Approximation (to be more accurate)
    if i != 0:
        # (a+b)/2 * dt
        thisEnergy = 1/2*(dataDict['P_battery'][i] + dataDict['P_battery'][i-1]) * (dataDict['t0'][i] - dataDict['t0'][i-1])
        thisEnergy_kWh = thisEnergy / 3600000   # convert to kWh
        dataDict['Energy Use'][i+1] = dataDict['Energy Use'][i] + thisEnergy_kWh
    
    return dataDict

### extraBatteryCalcs
## Battery thermals
def extraBatteryCalcs(dataDict, i):
    ########################################################################
    # Simple thermal calculations
    # P = I^2 * r - absolute of pack current to accout for regen also increasing pack temp
    dataDict['Dissipated Power'][i] = (abs(dataDict['Pack Current'][i]) / num_parallel_cells)**2 * single_cell_ir
    
    # Previous power out of cell calculation
    # cell_p_out = (air_tc)*(dataDict['Battery Temp'][i]-air_temp) + water_tc*(dataDict['Battery Temp'][i]-water_temp)

    # Calculate time interval
    dt = dataDict['t0'][i+1] - dataDict['t0'][i]

    # Power out of cell to heatsink
    cell_p_out = (dataDict['Battery Temp'][i] - dataDict['Heatsink Temp'][i]) / (thermal_resistance_SE * num_parallel_cells)

    # First temp calculations
    dataDict['Battery Temp'][i+1] = dataDict['Battery Temp'][i] + dt * (dataDict['Dissipated Power'][i] - cell_p_out) / (battery_heat_capacity)

    # Heatsink temp calculations
    # Split up calculation:
    calculation1 = (dataDict['Battery Temp'][i] - dataDict['Heatsink Temp'][i]) * num_series_cells / thermal_resistance_SE - air_tc * (dataDict['Heatsink Temp'][i] - air_temp)
    dataDict['Heatsink Temp'][i+1] = dataDict['Heatsink Temp'][i] + dt / (heatsink_mass * heatsink_cv) * calculation1

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

    # Plot 1)
    # Regen vs Distance
    row = 0; col = 0
    x_axis = "Distance (m)"
    y_axis = "Torque (Nm)"
    plotTitle = "Motor Torque vs Distance"
    # Convert the velocity from m/s to km/h
    # dataDict['v0'] = dataDict['v0'] * 3.6
    ax[row][col].plot(dataDict["r0"], dataDict["T_m"])       # plot the data
    plotDetails(x_axis, y_axis, plotTitle, ax[row][col])  # add the detaila

    # # Plot 2)
    # # Velocity vs Distance
    # row = 0; col = 1
    # x_axis = "Distance (s)"
    # y_axis = "Speed (km/h)"
    # plotTitle = "Speed vs Distance"
    # ax[row][col].plot(dataDict["r0"], dataDict["v0"])
    # plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

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

    # Plot 4)
    # Battery 
    row = 1; col = 1
    x_axis = "Distance (m)"
    y_axis = "Total Losses (kW)"
    plotTitle = "Drooped Total Losses vs Distance"
    ax[row][col].plot(dataDict['r0'], dataDict['Total Losses'] / 1000)
    # Will also plot a red line to show the minimum voltage
    # ax[row][col].plot(dataDict['t0'], np.ones_like(dataDict['t0']) * pack_min_voltage, 'r')
    plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

    return


#############################
# UNUSED:

# ### batteryPower
# ## NEW BATTERY FUNCTION
# def batteryPower(dataDict, i, ShaftTorque, MotorPower, TotalLosses, AMK_current, AMK_speeds):
#     # separate into regen/non-regen
#     if dataDict['T_m'][i] < 0:
#         if regen_on == "TRUE":
#             pass
#         else: 
#             # just set the output to ZERO
#             dataDict['Pack Current'][i] = 0
#             dataDict['P_battery'][i] = 0
#     else:
#         # determine current pulled from each motor
#         RPM_index = findClosestMatch(AMK_speeds, dataDict['w_m'][i])
#         Torque_index = findClosestMatch(ShaftTorque.iloc[RPM_index, :].to_list(), dataDict['T_m'][i])
#         # single_motor_current = AMK_current[Torque_index]
#         P_3phase = MotorPower.iloc[RPM_index, Torque_index]

#         # Power into the inverter = P_motor-to-invert / n_converter
#         P_intoInverter = P_3phase / n_converter

#         # Then losses along the bussing line
#         # Power from battery = Power into inverter
#         # Taking currently known voltage, determine current running through pack
#                 # Now I'm a bit stuck on how to calculate losses that involve current: IR losses and bussing losses - but they could be done separately
#                 # The question is... since we assume power stays constant through the inverter - how do we calculate current into the inverter? Anyway...
#                 # Cause that would allow us to calculate voltage-drooping
        
#         dataDict['P_battery'][i] = P_intoInverter * 4   # right now without losses
#         dataDict['Pack Current'][i] = dataDict['P_battery'][i] / pack_nominal_voltage

#         # four_motor_current = single_motor_current * 4       # to check and compare to the pack current

#         # # Add motor losses
#         # single_motor_losses = TotalLosses.iloc[RPM_index, Torque_index]

#         # # Motor power
#         # single_motor_power = single_motor_current * pack_nominal_voltage + single_motor_losses

#         # # Single inverter power
#         # single_inverter_power = single_motor_power / n_converter

#         # # Outputs from FOUR motors
#         # power_from_four = single_inverter_power * 4

#         # # Outputs from battery considering internal resistance (an estimate...)
#         # dataDict['Dissipated Power'][i] = total_pack_ir * four_motor_current**2

#         # # # Add calculation for bussing losses
#         # # # Calculate bussing in the constants section as: bussing length * resistivity / cross_sectional_area
#         # # bussing_losses = bussing_ir * four_motor_current**2

#         # # Include inverter losses
#         # dataDict['P_battery'][i] = power_from_four + dataDict['Dissipated Power'][i]
#         # dataDict['Pack Current'][i] = four_motor_current #dataDict['P_battery'][i] / pack_nominal_voltage
#         # # NOT SURE IF THIS LAST PART IS CORRECT, OR IF IT SHOULD BE four_motor_current

#         # # Checking to see if voltage-drooping is present in the motor current measurement
#         # if four_motor_current != 0:
#         #     dataDict['Drooped Voltage'][i] = dataDict['P_battery'][i] / four_motor_current
        
#         # # Record total losses
#         # dataDict['Total Losses'][i] = single_motor_losses * 4 + dataDict['Dissipated Power'][i]

#         # # dataDict['P_battery'][i] = dataDict['T_m'][i] * dataDict['w_m'][i] / n_converter
#         # # dataDict['Pack Current'][i] = dataDict['P_battery'][i] / pack_nominal_voltage

#     return dataDict