import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import csv
import json
import dynFunctions as dynF
import datetime
import pickle

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
max_current = num_parallel_cells * cell_max_current                         # A - Max current through pack
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
battery_heat_capacity = cell_cv*cell_mass                                # J/C
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

#####################################
# IMPORT DATASETS

print("Max Power Limit: %d W" % max_power)

####################
if motor_choice == "AMK":
    # AMK Motor
    # Import AMK Data
    in_json = AMKData
    in_json = open(in_json)
    in_json = in_json.read()
    AMK_dict = json.loads(in_json)

    # Converting list/dict storage into Dataframe and Array Storage
    ShaftTorque = pd.DataFrame(AMK_dict['ShaftTorque'])
    PowerFactor = pd.DataFrame(AMK_dict['PowerFactor'])
    TotalLosses = pd.DataFrame(AMK_dict['TotalLosses'])
    PeakTorqueSpeed = pd.DataFrame(AMK_dict['PeakTorqueSpeed']) # Just torque/speed
    ContTorqueSpeed = pd.DataFrame(AMK_dict['ContTorqueSpeed']) # Just torque/speed
    LineVoltageRMS = pd.DataFrame(AMK_dict['LineVoltageRMS'])
    MotorPower = pd.DataFrame(AMK_dict['MotorPower'])

    AMK_current = list(ShaftTorque.columns)
    AMK_current = np.array(AMK_current, dtype='float')
    AMK_speeds = list(ShaftTorque.index)
    AMK_speeds = np.array(AMK_speeds, dtype='float')

    print('AMK Chosen')

    if torque_speed == "Peak":
        TorqueSpeed = PeakTorqueSpeed
    elif torque_speed == "Continuous":
        TorqueSpeed = ContTorqueSpeed
    else:
        print("Incorrect torque/speed choice input")
        quit()

####################
else:
    # Emrax Motor
    # Convert .json to dictionary
    in_json = 'emrax_data.json'
    in_json = open(in_json)
    in_json = in_json.read()
    emrax_dict = json.loads(in_json)

    # Converting list/dict storage into Dataframe and Array storage
    MotorEfficiency = pd.DataFrame(emrax_dict['Motor Efficiency'])
    PeakTorqueSpeed = np.array(emrax_dict['PeakTorqueSpeed'])
    ContTorqueSpeed = np.array(emrax_dict['ContTorqueSpeed'])

    # Additional emrax data - also need a function handle for the Torque/Current Characteristic
    phases = 3
    pole_pairs = 10
    pm = emrax_dict['lambda_pm']

    # Function handle for managing torque --> Current changes (if necessary)
    torque_current = lambda I: phases / 2 * pole_pairs * pm * I

    print('Emrax chosen')

####################
# Track Data
# Finally read the track data
trackData = pd.read_csv(TRACK)

########################################
# PRE-ITERATION ANALYSIS

###################
# Track maximum velocities

b_vec = (trackData['Radius'] == 0) * 100000       # Create boolean vector for input data - and update the radius to be VERY large on straights
trackData['Radius'] = trackData.loc[:,'Radius'] + b_vec   # Then we update the radius vector

# And now... add a column to the dataframe with the maximum speed in each section (this is kinda like doing vector operations in MATLAB)
trackData = trackData.assign(MaxVelocity = np.sqrt(a_centrip * trackData['Radius']))    # Could make this more accurate by basing it off of down force at a specific time and speed, but idk...

# Vector length:
num_intervals = int(trackData.loc[trackData.index[-1], 'Cumulative Length'] / delta_d)

###################
# Braking Iteration

# Initialize a dictionary to store all of the braking informtion
TrackLength = int(trackData.shape[0]) # Provides the length of a column of the df

brakeDict = dict.fromkeys(['Distance', 'Speed'], None)

# Dataframe parameters:
dictList = ['Time','Speed','Distance']

# Iterate through the track loop to determine the braking data at each point
for i in range(0, TrackLength):
    # Find sector length
    sectorLength = trackData.loc[trackData.index[i], 'Section Length']

    # Find sector vector length
    vectorLength = int(sectorLength / delta_d)

    # Initialize brakeSector_dict
    brakeSector_dict = dict.fromkeys(dictList)
    
    # add empty zero vectors...
    for k in range(0, len(dictList)):
        brakeSector_dict[dictList[k]] = np.zeros(vectorLength)

    # Initialize values in dict
    brakeSector_dict['Time'][vectorLength-1] = 0
    brakeSector_dict['Distance'][vectorLength-1] = trackData.loc[trackData.index[i], "Cumulative Length"]

    # Initialize the Max Velocity (Taking into account edge cases)
    if i != (TrackLength - 1):
        brakeSector_dict['Speed'][vectorLength - 1] = trackData.loc[trackData.index[i+1], 'MaxVelocity']
    else:
        brakeSector_dict['Speed'][vectorLength - 1] = trackData.loc[trackData.index[0], 'MaxVelocity']

    # Fill up the distance, time, and velocity from the back of the dataframe
    for j in range(0, vectorLength-1):
        # Iteration for distance, time, and velocity
        # First: solve for delta time:
        poly_coeffs = np.array([1/2, brakeSector_dict['Speed'][vectorLength - j - 1], -delta_d])          # coefficients: p[0] * x^n + ... + p[n]
        dt = np.roots(poly_coeffs)[1]                            # This is the delta_t
        brakeSector_dict['Time'][vectorLength - j - 2] = brakeSector_dict['Time'][vectorLength - j - 1] + dt

        # then for v1
        brakeSector_dict['Speed'][vectorLength - j - 2] = brakeSector_dict['Speed'][vectorLength - j - 1] + brake_decel * dt

        # Then for distance
        brakeSector_dict['Distance'][vectorLength - j - 2] = brakeSector_dict['Distance'][vectorLength - j - 1] - delta_d

    # FINALLY: save the dataframe into the brakeDict
    brakeDict['Speed'] = np.append(brakeDict['Speed'], brakeSector_dict['Speed'])
    brakeDict['Distance'] = np.append(brakeDict['Distance'], brakeSector_dict['Distance'])
    # brakeDict[sectors[i]] = brakeSector_dict

# Drop first row
brakeDict['Distance'] = np.delete(brakeDict['Distance'], 0)
brakeDict['Speed'] = np.delete(brakeDict['Speed'], 0)

##########################################################
# BEGIN ITERATION

# Begin the iteration:
# CURRENT TIME
currentTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Vectors for each set of data
# Create a dictionary of values
headers = ['v0',                    # velocity vector (m/s)
           'r0',                    # distance vector (m)
           't0',                    # time vector (s)
           'w_wh',                  # wheel angular velocity (rad/s)
           "w_m",                   # motor angular velocity (rpm)
           "T_m",                   # motor torque (Nm)
           "Motor Power Loss",      # Power Losses from Motor (W)
           "Motor Energy Loss",     # Total Energy Loss from Motor (J)
           "Inverter Power Loss",   # Power Losses from Inverter (W)
           "Inveter Energy Loss",   # Total Energy Loss from Inverter (J)
           "T_a",                   # Axel torque (Nm)
           "F_trac",                # traction force (N)
           "F_drag",                # drag force (N)
           "F_down",                # down force (N)
           "F_RR",                  # rolling resistance (N)
           "F_net_tan",             # net force in the tangential direction (N)
           "a_tan0",                # tangential acceleration (m/s^2)
           "P_battery",             # battery power (kW)
           "Pack Voltage",          # battery voltage (V)
           "Pack Capacity",         # Battery capacity (Ah)
           "Pack Current",          # Battery pack current (A)
           "Energy Use",            # energy use over time (kWh)
           "SoC Capacity",          # state of charge - capacity based (%)
           "Cell Qgen",             # Power dissipated from batteries due to internal resistance (W)
           "Cell Qout",             # Power transferred OUT of cell (W)
           "Cell Net Q",            # Net power generated by cell (Qgen - Qtransferred) (W)
           "Cell Total Gen",       # = Cell Qgen * delta_t (J)
           "Cell Total Qout",       # = Cell Qout * delta_t (J)
           "Cell Net Heat",         # = Cell Net Q * delta_t (J)
           "Battery Temp",          # Temperature of battery pack (C)
           "Drooped Voltage",       # To determine whether voltage drops are considered with the battery power / motor current inconsistency problem... (V)
           "Total Losses",          # Total Losses with motor and battery (kW)
           "Total Losses NRG",      # Total Losses from motor and battery - energy (kWh)
           "Heatsink Temp"]         # Temperature of the heat sink (C)
dataDict = dict.fromkeys(headers)

# add empty zero vectors
for i in range(0, len(headers)):
    dataDict[headers[i]] = np.zeros(num_intervals)

# Add some starting values
dataDict['Pack Capacity'][0] = pack_capacity_initial
dataDict['SoC Capacity'][0] = initial_SoC
dataDict['Battery Temp'][0] = batteryTemp0
dataDict['Heatsink Temp'][0] = heatsink_temp_0
dataDict['Pack Voltage'][0] = pack_max_voltage  # Without SoC prediction, we'll set this to be nominal voltage ALL THE WAY!! with a np.ones * nom voltage

# DEBUG CONSTANTS
longitudinal_traction_limits = 0
max_speed_limits = 0
power_limits = 0
current_limits = 0
braking_limits = 0
regen_current_limits = 0
laps_completed = 0

# CALCULATIONS
for i in range(0, num_intervals-1):
    # INITIAL CALCULATIONS

    # Calculates the fastest next speed and the smallest possible time
    dataDict, longitudinal_traction_limits = dynF.fastestNextSpeed(dataDict, TorqueSpeed, i, longitudinal_traction_limits)

    #########################################  
    # TRACTION CALCULTIONS

    # Now, check to find the maximum actually possible speed based on traction considerations
    v_max = dynF.findMaxSpeed(trackData, dataDict, i)

    # Now determine whether we exceed the maximum speed, and if so, recalculate the possible values
    if dataDict['v0'][i+1] > v_max:
        # print('Above max speed - iteration %d' % i)
        dataDict, max_speed_limits = dynF.limit_max_speed(dataDict, v_max, i, max_speed_limits)

    # Add the braking
    dataDict, braking_limits = dynF.braking(brakeDict, dataDict, i, braking_limits)

    ###########################################
    # BATTERY CALCULATIONS

    # Determine battery power used during the race
    dataDict, regen_current_limits = dynF.batteryPower(dataDict, i, ShaftTorque, MotorPower, AMK_speeds, regen_current_limits, AMK_current)

    # Add safety checks on the battery here
    dataDict, power_limits, current_limits = dynF.batteryChecks(dataDict, i, AMK_current, AMK_speeds, ShaftTorque, power_limits, TotalLosses, MotorPower, current_limits)

    # Additional Battery Calculations including SoC approximation and temperature
    dataDict = dynF.extraBatteryCalcs(dataDict, i)

    # Energy calculations
    dataDict = dynF.energyConsumed(dataDict, i)

    # Update user on number of laps completed
    LINE_CLEAR = '\x1b[2K'

    if laps_completed != round(dataDict['r0'][i] / 1000, 0):
        laps_completed = round(dataDict['r0'][i] / 1000, 0)
        print(LINE_CLEAR, end = '\r')
        print("Laps Completed: %d" % laps_completed, end='\r')

    # if dataDict['r0'][i] > 36.4:
    #     print("debug distance: %.2f m" % dataDict["r0"][i])
    #     print("debug current: %.3f A" % dataDict["Pack Current"][i])
    #     print("debug power: %.3f kW" % (dataDict["P_battery"][i]/1000))

# Energy use
total_energy = dataDict['Energy Use'][-1]
total_energy_loss = dataDict['Total Losses NRG'][-1]

# Determination of maximum power
dataDict['P_battery'] = dataDict['P_battery'] / 1000        # convert to kW
maxPower = max(dataDict['P_battery'])
averagePower = np.mean(dataDict['P_battery'])

# Convert velocities to km/h
dataDict['v0'] = dataDict['v0'] * 3.6            # convert to km/h

####################################
# OUTSIDE FOR-LOOP CALCULATIONS

# Print relevant outputs to terminal
print('Energy Used (This Sim): ' + str(total_energy) + ' kWh')
if track_choice == "Autocross":
    print('Energy Used (22 Laps - Endurance): %.6f kWh' % (dataDict['Energy Use'][-1] * 22))
print("Total Energy Lost (This Sim): %.3f kWh" % total_energy_loss)
print("Max Power (This Sim): " + str(maxPower) + " kW")
print("Avg Power (This Sim): " + str(averagePower) + " kW")
print("Car Mass: " + str(mass) + " kg")
print("Lap Time: " + str(dataDict['t0'][-1] / numLaps) + " s")
print("Total Time: " + str(dataDict['t0'][-1] / 60) + " mins")
print("Average pack current: %.6f A" % np.mean(dataDict['Pack Current']))
# print("Final SoC(c): ", dataDict['SoC Capacity'][-1], "%")

# Write all terminal output to a *.txt file
summaryOutPath = summaryOutPath + cell_choice + "\\"
outfileName =  currentTime + "_" + cell_choice + '_' + track_choice + ".txt"
summaryOutPath = summaryOutPath + outfileName
with open(summaryOutPath, 'w') as textFile:
    textFile.write('Energy Used (This Sim): ' + str(total_energy) + ' kWh\n')
    if track_choice == "Autocross":
        textFile.write('Energy Used (22 Laps - Endurance): %.6f kWh\n' % (dataDict['Energy Use'][-1] * 22))
    textFile.write("Total Energy Lost (This Sim): %.3f kWh\n" % total_energy_loss)
    textFile.write("Max Power (This Sim): " + str(maxPower) + " kW\n")
    textFile.write("Avg Power (This Sim): " + str(averagePower) + " kW\n")
    textFile.write("Average pack current: %.3f A\n" % np.mean(dataDict['Pack Current']))
    textFile.write("Total Heat Generated from Batteries: %.3f J\n" % dataDict["Cell Total Gen"][-1])
    textFile.write("Total Heat Removed from Batteries: %.3f J\n" % dataDict['Cell Total Qout'][-1])
    textFile.write("Total Net Generated Heat from Batteries: %.3f J\n" % dataDict['Cell Net Heat'][-1])
    textFile.write("Car Mass: " + str(mass) + " kg\n")
    textFile.write("Lap Time: " + str(dataDict['t0'][-1] / numLaps) + " s\n")
    textFile.write("Total Time: " + str(dataDict['t0'][-1] / 60) + " mins\n")
    textFile.write("Longitudinal traction limits: " + str(longitudinal_traction_limits) + "\n")
    textFile.write("Max Speed Limits: " + str(max_speed_limits) + "\n")
    textFile.write("Power Limits: " + str(power_limits) + "\n")
    textFile.write("Current Limits: " + str(current_limits) + "\n")
    textFile.write("Braking Limits: " + str(braking_limits) + "\n")
    textFile.write("Regen Current Limits: %d\n" % regen_current_limits)

# Create plots
fig = dynF.plotData(dataDict, currentTime)
figTitle = currentTime + "_" + cell_choice + '_' + track_choice + ".png"
outputPlotPath = outputPlotPath + "\\" + cell_choice + "\\" + figTitle
plt.savefig(outputPlotPath)
fig.clear(True)

# Now I want to write all the columns to a dictionary and then input it into a dataframe - since it's easier to do column-wise
dfData = pd.DataFrame(dataDict)
dfData.dropna(inplace = True)

if track_choice == "Autocross":
    # Drop data to 10th of output size
    dfData = dfData.iloc[::10]
elif track_choice == "Endurance":
    # Drop data size to a 100th of output size
    dfData = dfData.iloc[::100]
elif track_choice == "SkidPad":
    # Drop data to 1/2 of output size
    dfdata = dfData.iloc[::2]

# Don't change the acceleration output

# Write to csv
dfData.to_csv(fullDataOutPath, index=False)

print("Completed")

###################################################
# OPTIMUM LAP VELOCITY PROFILE COMPARISON

if track_choice == "Autocross":

    # Optimum Lap Comparison
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Analyze current data
    time = dataDict['t0']
    velocity = dataDict['v0']
    distance = dataDict['r0']

    # Import actual csv
    infile = OLData
    dfO = pd.read_csv(infile, header = [0])
    dfO = dfO.drop(index = 0)
    distanceO = dfO.loc[:,'elapsedDistance'].to_numpy(dtype = float)
    timeO = dfO.loc[:,'elapsedTime'].to_numpy(dtype = float)
    velocityO = dfO.loc[:,'speed'].to_numpy(dtype = float)

    # plot the graphs on top of each other
    plt.plot(distanceO, velocityO, 'b')
    plt.plot(distance, velocity, 'r')
    plt.xlabel('Distance (m)')
    plt.ylabel('Velocity (km/h)')
    plt.title('Velocity vs Distance over 1 Endurance Lap')
    plt.legend(['Optimum Lap Data', 'Python Lap Simulation Data'])
    plt.grid(True)

    OLFileName = currentTime + "_" + cell_choice + '_' + track_choice + "_OLPlot.png"
    OLPlotPath = OLPlotPath + "\\" + cell_choice + "\\" + OLFileName
    plt.savefig(OLPlotPath)