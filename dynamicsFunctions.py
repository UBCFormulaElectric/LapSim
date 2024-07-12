import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import csv

############################################################################
# Mallory Moxham - UBC Formula Electric - July 2024
############################################################################

# USER-INPUT CONSTANTS
# STRING CONSTANTS
string_constants = 3
TRACK = None
regen_on = None                 # True/False - Regen on or off

# TRACK CONSTANTS
numLaps = None

# CAR CONSTANTS
no_cells_car_mass = None                     # kg - CAR MASS
Af = None                       # m^2 - FRONTAL AREA
Al= None                       # m^2 - WING ELEMENT AREA FROM TOP DOWN
mu_rr = None                    # COEFFICIENT OF ROLLING RESISTANCE

# BATTERY CONSTANTS
# DEPENDS ON STARTING CONDITIONS
initial_SoC = None              # % - INITIAL STATE OF CHARGE
starting_voltage = None         # V - INITIAL PACK VOLTAGE
capacity0 = None                # Ah - INITIAL PACK CAPACITY
# DEPENDS ON BATTERY CHOICE
max_capacity = None             # Ah
n_converter = None              # converter efficiency
cell_max_voltage = None         # V - MAX CELL VOLTAGE
cell_min_voltage = None         # V - MIN CELL VOLTAGE
cell_nominal_voltage = None
num_series_cells = None         # NUMBER OF SERIES ELEMENTS
num_parallel_cells = None       # NUMBER OF PARALLEL CELLS
single_cell_ir = None           # Ohms - CELL INTERNAL RESISTANCE
max_CRate = None                # Max C-Rate
cell_mass = None                # Mass of single cell
battery_cv = None               # Specific heat capacity of battery
# !!!
cell_water_area = None          # m^2 - WATER COOLING SURFACE OF CELL
cell_aux_factor = None          # kg/kWh - SEGMENT AUXILLARY MASS/ENERGY

# MOTOR CONSTANTS
motor_choice = None              # emrax / AMK - choose your motor

# TRACTION CONSTANTS
max_speed_kmh = None            # km/h - MAX SPEED
traction_speed = None           # km/h - MAX SPEED AROUND RADIUS IN TRACTION TEST
traction_radius = None          # m - RADIUS OF TRACTION TEST
mu_longitudinal = None          # Coefficient of longitudinal friction
mu_lateral = None               # Coefficient of lateral friction
brake_decel = None              # m/s^2 - BRAKING DECELERATION RATE
Cl = None                       # Lift coefficient
Cd = None                       # Draf coefficient

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

######################################################################
# This allows an external user to have control over the constants without touching the code

# Open .csv and take constants as input:
filename = "LapSimConstants.csv"

# Open the file
with open(filename, 'r', newline='') as infile:
    reader = csv.reader(infile)
    dataList = list(reader)
    dataList.pop(0)             # Remove title row

    # convert to array
    dataArray = np.array(dataList)

    # Take out the valuable columns and convert to floats as necessary
    value_name = dataArray[2:,0]
    value = dataArray[2:,2]
    value = np.asarray(value, dtype = float)

    track_name = dataArray[0,0]
    track = dataArray[0,2]

    regen_name = dataArray[1,0]
    regen = dataArray[1,2]

# Now create variables for everything
for x, y in zip(value_name, value):
    globals()[x] = y

globals()[track_name] = track
globals()[regen_name] = regen

###################################################################################
# CALCULATED CONSTANTS

# CONSTANTS - SHOULD NOT NEED CHANGING
delta_d = 0.01           # distance interval - m
g = 9.81                # m/s^2
# Motor
motor_choice = "AMK"
if motor_choice == 'AMK':
    GR = 14.33              # Gear Ratio - AMK
else:
    GR = 4.2                # Gear Ratio - emrax
wheel_diameter = 18 * 0.0254    # m
wheel_radius = wheel_diameter / 2
rho_air = 1.204         # air density: kg / m^3
v_air = 0               # air velocity: m/s
radsToRpm = 1 / (2 * math.pi) * 60    # rad/s --> rpm
max_power = 80000       # kW

# Battery Pack - Calculated Values
num_cells = num_series_cells * num_parallel_cells
pack_nominal_voltage = cell_nominal_voltage * num_series_cells # V
pack_max_voltage = cell_max_voltage * num_series_cells # V
total_pack_ir = single_cell_ir / num_parallel_cells * num_series_cells  # ohms
# !!! Total known energy is approximately SoC * nominal voltage * max capacity
knownTotalEnergy = initial_SoC * capacity0 * pack_nominal_voltage / 1000  # kWh
pack_min_voltage = cell_min_voltage * num_series_cells  # V

# !!!
# Car Mass - Calculated Values
total_cell_mass = cell_mass*num_cells # kg
cooled_cell_mass = total_cell_mass*(1 + air_factor_m + water_factor_m) # kg
cell_aux_mass = cell_aux_factor*(capacity0 * pack_nominal_voltage / 1000) # kg
mass = no_cells_car_mass + cooled_cell_mass + cell_aux_mass + heatsink_mass # kg

# !!! 
# Thermals - Calculated Values
battery_heat_capacity = battery_cv*cell_mass # J/C
air_tc = air_htc*heatsink_air_area  # W/C
water_tc = water_htc*cell_water_area # W/C
air_thermal_resistance = 1 / air_tc  # K/W
heatsink_temp_0 = air_temp           # C
batteryTemp0 = air_temp             # C - starting temperature of battery pack (may change if necessary)

# Traction Constants
# at 30 km/h, we travelled around a 5 m radius circle
a_centrip = (traction_speed * 1000 / 3600)**2 / traction_radius      # v^2 / r (convert to m/s)
test_mass = 225                             # kg - car mass used in testing
F_friction = test_mass * a_centrip          # calculate the friction force
mu_lateral = F_friction / (test_mass * g)         # calculate the tire friction coefficient

# Some more arbitrary speed measurements
max_speed = max_speed_kmh / 3.6             # m/s
max_traction_force = mass * g * mu_longitudinal   # N - max force in LONGITUDINAL DIRECTION
F_friction = mu_lateral * mass * g                # friction force based on the evaluated car mass.


#################################################################################################
# FUNCTIONS
#################################################################################################

def quad_formula(a, b, c):
    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        root1 = 0
        root2 = 0
    else:
        root1 = (-b + discriminant**(1/2)) / (2*a)
        root2 = (-b - discriminant**(1/2)) / (2*a)

    return [root1, root2]

# function for finding closest match (rather than searchsorted)
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

# Solve for next time:
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

    # if i == 4157:
    #     debug = np.roots(poly_coeffs)
    #     print('wait')

    return dt

# Round down for floating point inaccuracies
def round_nearest(x, a):
    return round(x / a) * a

# the initial calculations to determine a speed and distance
def fastestNextSpeed(dataDict, PeakTorqueSpeed, i):
        # angular frequency of wheel: w_wh
    dataDict['w_wh'][i] = dataDict['v0'][i] / wheel_radius

    # angular frequency of motor: w_m - also convert from rad/s to rpm
    dataDict['w_m'][i] = dataDict['w_wh'][i] * GR * radsToRpm

    # solve for motor torque: T_m
    index = findClosestMatch(PeakTorqueSpeed.loc[:,'Speed Peak'].to_list(), dataDict['w_m'][i])
    dataDict['T_m'][i] = PeakTorqueSpeed.iloc[index, 1]

    # axel torque: T_a
    dataDict['T_a'][i] = dataDict['T_m'][i] * GR

    # Traction force: F_trac
    dataDict['F_trac'][i] = 2 * dataDict['T_a'][i] / wheel_radius  # Traction force from FOUR motors
    # NOT -> F_trac[i] = T_a[i] / (2 * wheel_radius)
    # MULTIPLIED BY TWO FOR 4WD!!

    # Down force
    dataDict['F_down'][i] = 1/2 * rho_air * Cl * Al * dataDict['v0'][i]

    # Determine max traction force from wheels: F_max = mu * F_normal = mu * (F_down + car weight force) 
    max_traction_force = mu_longitudinal * (dataDict['F_down'][i] + mass * g)

    # Now determine if the car is traction limited - if so, reduce torque applied to wheels.
    if dataDict['F_trac'][i] > max_traction_force:
        dataDict['F_trac'][i] = max_traction_force

        dataDict['T_a'][i] = wheel_radius * dataDict['F_trac'][i]

        dataDict['T_m'][i] = dataDict['T_a'][i] / GR

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

    return dataDict

# determining the max speed requirement
def findMaxSpeed(trackData, dataDict, i):
    trackLocation = np.searchsorted(trackData['Cumulative Length'].values, dataDict['r0'][i])

    v_max = trackData['MaxVelocity'][trackLocation]

    return v_max

# Based on the new max speed, what are our new variables
def limit_max_speed(dataDict, v_max, i):
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
    dataDict['T_a'][i] = dataDict['F_trac'][i] * wheel_radius

    # motor torque
    dataDict['T_m'][i] = dataDict['T_a'][i] / GR

    # w_m, w_wh, v0 stay the same

    # New time also changes:
    dt = nextTime(dataDict, i)
    dataDict['t0'][i+1] = dataDict['t0'][i] + dt

    return dataDict

# Function to incorporate and calculate for braking iterativelys
def braking(brakeDict, dataDict, i):
    # Find location
    trackLocation = findClosestMatch(brakeDict['Distance'], dataDict['r0'][i])

    if trackLocation > len(brakeDict['Distance']) - 1:
        trackLocation = trackLocation - 1

    if dataDict['v0'][i+1] > brakeDict['Speed'][trackLocation]:
        # Reset the speed to the braking speed if necessary
        dataDict['v0'][i+1] = brakeDict['Speed'][trackLocation]

        # Back calculate additional parameters - TAKE NOT LATER IF REGEN IS ON OR NOT
        # IF REGEN OFF, THEN ENSURE THAT NEGATIVE TORQUE --> NO BATTERY POWER
        # Now determine the required acceleration at this point.
        dataDict['a_tan0'][i] = (dataDict['v0'][i+1]**2 - dataDict['v0'][i]**2) / (2 * delta_d)
        # dataDict['a_tan0'][i] = (dataDict['v0'][i+1] - dataDict['v0'][i]) / dt

        # Now, what is the net force
        dataDict['F_net_tan'][i] = mass * dataDict['a_tan0'][i]

        # Now based on the net force, what is the traction force sent to the wheels
        dataDict['F_trac'][i] = dataDict['F_net_tan'][i] + (dataDict['F_drag'][i] + dataDict['F_RR'][i])

        # axel torque
        dataDict['T_a'][i] = dataDict['F_trac'][i] * wheel_radius

        # motor torque (SHOULD BE NEGATIVE!!)
        dataDict['T_m'][i] = dataDict['T_a'][i] / GR

        # w_m, w_wh, v0 stay the same

        # New time also changes:
        dt = nextTime(dataDict, i)
        dataDict['t0'][i+1] = dataDict['t0'][i] + dt

    return dataDict

# NEW BATTERY FUNCTION
def batteryPower(dataDict, i, ShaftTorque, PowerFactor, TotalLosses, AMK_current, AMK_speeds):
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
        single_motor_current = AMK_current[Torque_index]
        four_motor_current = single_motor_current * 4       # to check and compare to the pack current

        # Add motor losses
        single_motor_losses = TotalLosses.iloc[RPM_index, Torque_index]

        # Motor power
        single_motor_power = single_motor_current * pack_nominal_voltage + single_motor_losses

        # Single inverter power
        single_inverter_power = single_motor_power / n_converter

        # Outputs from FOUR motors
        power_from_four = single_inverter_power * 4

        # # Outputs from battery considering internal resistance (an estimate...)
        # ir_losses = total_pack_ir * four_motor_current**2

        # Include inverter losses
        dataDict['P_battery'][i] = power_from_four # + ir_losses
        dataDict['Pack Current'][i] = dataDict['P_battery'][i] / pack_nominal_voltage
        # NOT SURE IF THIS LAST PART IS CORRECT, OR IF IT SHOULD BE four_motor_current

    return dataDict

# Battery safety checks
def batteryChecks(dataDict, i, AMK_current, AMK_speeds, ShaftTorque):
    # Compare power limit with overcurrent fault:
    max_current = num_parallel_cells * max_CRate * max_capacity
    max_current_power_limited = max_power / pack_nominal_voltage

    # if the power limit is more conservative
    if max_current_power_limited < max_current:
        # Check for over 80 kW
        if dataDict['P_battery'][i] > max_power:
            dataDict['P_battery'][i] = max_power

            # now back calculate additional values
            dataDict['Pack Current'][i] = max_power / pack_nominal_voltage

            # Determine resulting max torque
            current_index = findClosestMatch(AMK_current, dataDict['Pack Current'][i])
            RPM_index = findClosestMatch(AMK_speeds, dataDict['w_m'][i])
            dataDict['T_m'][i] = ShaftTorque.iloc[RPM_index, current_index]

            ###############################################################################################
            # Now the rest of the values
            # axel torque: T_a
            dataDict['T_a'][i] = dataDict['T_m'][i] * GR

            # Traction force: F_trac
            dataDict['F_trac'][i] = 2 * dataDict['T_a'][i] / wheel_radius  # Traction force from FOUR motors
            # NOT -> F_trac[i] = T_a[i] / (2 * wheel_radius)
            # MULTIPLIED BY TWO FOR 4WD!!

            # Down force
            dataDict['F_down'][i] = 1/2 * rho_air * Cl * Al * dataDict['v0'][i]

            # Determine max traction force from wheels: F_max = mu * F_normal = mu * (F_down + car weight force) 
            max_traction_force = mu_longitudinal * (dataDict['F_down'][i] + mass * g)

            # Now determine if the car is traction limited - if so, reduce torque applied to wheels.
            if dataDict['F_trac'][i] > max_traction_force:
                dataDict['F_trac'][i] = max_traction_force

                dataDict['T_a'][i] = wheel_radius * dataDict['F_trac'][i]

                dataDict['T_m'][i] = dataDict['T_a'][i] / GR

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
            ##############################################################################################
    
    # If the current limit is more conservative
    else:
        if dataDict['Pack Current'][i] >= max_current:
            dataDict['Pack Current'][i] = max_current

            # now back calculate rest of values
            dataDict['P_Battery'][i] = max_current * pack_nominal_voltage

            # Determine torque
            current_index = findClosestMatch(AMK_current, max_current)
            RPM_index = findClosestMatch(AMK_speeds, dataDict['w_m'][i])
            dataDict['T_m'][i] = ShaftTorque.iloc[RPM_index, current_index]

            ###############################################################################################
            # Now the rest of the values
            # axel torque: T_a
            dataDict['T_a'][i] = dataDict['T_m'][i] * GR

            # Traction force: F_trac
            dataDict['F_trac'][i] = 2 * dataDict['T_a'][i] / wheel_radius  # Traction force from FOUR motors
            # NOT -> F_trac[i] = T_a[i] / (2 * wheel_radius)
            # MULTIPLIED BY TWO FOR 4WD!!

            # Down force
            dataDict['F_down'][i] = 1/2 * rho_air * Cl * Al * dataDict['v0'][i]

            # Determine max traction force from wheels: F_max = mu * F_normal = mu * (F_down + car weight force) 
            max_traction_force = mu_longitudinal * (dataDict['F_down'][i] + mass * g)

            # Now determine if the car is traction limited - if so, reduce torque applied to wheels.
            if dataDict['F_trac'][i] > max_traction_force:
                dataDict['F_trac'][i] = max_traction_force

                dataDict['T_a'][i] = wheel_radius * dataDict['F_trac'][i]

                dataDict['T_m'][i] = dataDict['T_a'][i] / GR

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
            ##############################################################################################

    return dataDict

# Process for this code block:
# Check for the following:
# Over 80 kW power from battery
# Over max current from battery
# Under-voltage fault - to add later
# Over-temperature fault - to add later

# Process for back calcs:
# power
# Power to battery

# Energy consumed
def energyConsumed(dataDict, i):

    # Add up energy over time to get an approximation
    thisEnergy = dataDict['P_battery'][i] * (dataDict['t0'][i+1] - dataDict['t0'][i])
    thisEnergy_kWh = thisEnergy / 3600000   # convert to kWh
    dataDict['Energy Use'][i+1] = dataDict['Energy Use'][i] + thisEnergy_kWh
    
    return dataDict

# Battery thermals
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

# Plot details to make my life cleaner :))
def plotDetails(x_axis, y_axis, plotTitle, ax):
    ax.set_title(plotTitle)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.grid(True)

    return

# Plot the data
def plotData(dataDict):
    # Plot the data in this function
    ROWS = 2
    COLS = 2
    FIGWIDTH = 20
    FIGHEIGHT = 12

    # create subplots
    fig, ax = plt.subplots(ROWS, COLS, figsize=(FIGWIDTH, FIGHEIGHT))
    supTitle = "Point Mass Vehicle Simulation - " + TRACK.replace(".csv","")
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

    # Plot 2)
    # Velocity vs Distance
    row = 0; col = 1
    x_axis = "Distance (s)"
    y_axis = "Speed (m/s)"
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
    # Battery Current vs. Time
    row = 1; col = 1
    x_axis = "Distance (m)"
    y_axis = "Battery Current (A)"
    plotTitle = "Battery Current vs Distance"
    ax[row][col].plot(dataDict['r0'], dataDict['Pack Current'])
    # Will also plot a red line to show the minimum voltage
    # ax[row][col].plot(dataDict['t0'], np.ones_like(dataDict['t0']) * pack_min_voltage, 'r')
    plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

    figTitle = "Point Mass Vehicle Simulation_" + TRACK.replace(".csv","") + ".png"
    plt.savefig(figTitle)

    return

####################
# OLD FUNCTIONS
####################

# # BATTERY CALCULATION
# # From given traction force, calculates power draw from battery
# def batteryPower(dataDict, i):
#     P_wheel = dataDict['F_trac'][i] * dataDict['v0'][i]                       # Both wheel power
#     # P_wheel2 = dataDict['w_m'][i]  / radsToRpm * dataDict['T_m'][i]           # These are both identical

#     # n_transmission calculation
#     n_transmission = motorEfficiency(dataDict['w_m'][i])
    
#     # !!!
#     P_motors = 2* P_wheel / n_transmission
#     P_motorloss = 2* A * dataDict['w_m'][i]**2 + B * dataDict['w_m'][i] + C      # Motorloss

#     # The traction force is for BOTH motors and both wheels, but we have two motors, so twice the loss
#     P_converter = P_motors + P_motorloss
#     dataDict['P_battery'][i] = P_converter / n_converter
#     P_battery_debug = dataDict['P_battery'][i]

#     return dataDict

# # trapezoidal approximation with energy as result
# def trapezoidApprox(P_battery):
#     trapezoidal_vector = 2 * np.ones(num)
#     trapezoidal_vector[0] = 1
#     trapezoidal_vector[-1] = 1
#     almost_energy = dt / 2 * trapezoidal_vector * P_battery
#     energy = np.cumsum(almost_energy) / 3600000     # convert to kWh
#     totalEnergy = energy[-1]

#     return [energy, totalEnergy]

# # Recursive algorithm for finding SoC in lookup table - should be somewhat faster than looking through all of the data
# # could have just used np.searchsorted - however, I did this for my own learning so yayyy
# def SoClookup(SoC, thisCapacity):
#     if len(SoC) == 1:           # Base Case
#         return SoC[0]
#     else:                       # Recursive Case
#         mid = len(SoC) // 2     # Floor division
#         if SoC[mid] > thisCapacity:
#             return SoClookup(SoC[mid:], thisCapacity)
#         elif SoC[mid] < thisCapacity:
#             return SoClookup(SoC[:mid], thisCapacity)
#         else:
#             return SoC[mid]      # Secondary base case

# # Braking Considerations
# def batteryBrakeAndRegen(dataDict, i):
#     # based on current distance, search for braking info at that distance
#     distanceIndex = np.searchsorted(distanceTravelled, dataDict['r0'][i])
#     brake = brakePosition[distanceIndex]

#     # Now determine new battery parameters as suggested by optimum lap
#     # determine engine power at that distance
#     engine_w = engine_w_vector[distanceIndex]
#     engine_T = engine_T_vector[distanceIndex]

#     P_motors = engine_w * engine_T / radsToRpm            # motor (engine) power
#     P_motorloss = A * engine_w**2 + B * engine_w + C      # Motorloss

#     # The traction force is for BOTH motors and both wheels, but we have two motors, so twice the loss
#     P_converter = P_motors + P_motorloss
#     P_battery = P_converter / n_converter

#     # set battery power to be zero at that point and then determine regen
#     if brake == 0:
#         dataDict['P_battery_OL'][i] = P_battery
#     else:
#         dataDict['P_battery'][i] = 0

#         dataDict['P_battery_regen'][i] = -P_battery     # Note that regen will be shown as negative

#     return dataDict

# # More battery calculations
# def batteryPackCalcs(dataDict, i):
#     # Determine pack current, but first...

#     # Check if we are using regen
#     if regen_on == "TRUE":
#         if dataDict['P_battery'][i] == 0:
#             # Note that the effect of the total pack internal resistance is included to make the calculation more accurate
#             current = quad_formula(total_pack_ir, pack_nominal_voltage, -1 * dataDict['P_battery_regen'][i])
#             dataDict['Pack Current'][i] = max(current)      # CHECK THIS PLEASE!!
#         else:
#             # Note that the effect of the total pack internal resistance is included to make the calculation more accurate
#             current = quad_formula(total_pack_ir, pack_nominal_voltage, -1 * dataDict['P_battery'][i])
#             dataDict['Pack Current'][i] = max(current)      # ignore the negative result from the quadratic formula
#     else: 
#         # Note that the effect of the total pack internal resistance is included to make the calculation more accurate
#         current = quad_formula(total_pack_ir, pack_nominal_voltage, -1 * dataDict['P_battery'][i])
#         dataDict['Pack Current'][i] = max(current)      # ignore the negative result from the quadratic formula
    
#     if dataDict['Pack Current'][i] > num_parallel_cells * max_CRate * max_capacity:
#         dataDict['Pack Current'][i] = num_parallel_cells * max_CRate

#         dataDict['P_battery'][i] = pack_nominal_voltage * dataDict['Pack Current'][i]

#         P_converter = n_converter * dataDict['P_battery'][i]

#         P_motorloss = 2* A * dataDict['w_m'][i]**2 + B * dataDict['w_m'][i] + C      # Motorloss

#         P_motors = P_converter - P_motorloss

#         # n_transmission calculation
#         n_transmission = motorEfficiency(dataDict['w_m'][i])

#         P_wheel = n_transmission * P_motors / 2

#         dataDict['T_m'][i] = P_wheel / (dataDict['w_m'][i] / radsToRpm)

#     # Determine Ahr lost at this current based on time interval dt
#     Ahr_difference = dataDict['Pack Current'][i] * dt / 3600

#     # Determine new capacity
#     dataDict['Capacity'][i+1] = dataDict['Capacity'][i] - Ahr_difference

#     # Determine new SoCc
#     dataDict['SoC Capacity'][i+1] = dataDict["Capacity"][i+1] / capacity0 * 100

#     ########################################################################
#     # Simple thermal calculations
#     # P = I^2 * r - absolute of pack current to accout for regen also increasing pack temp
#     dataDict['Dissipated Power'][i] = (abs(dataDict['Pack Current'][i]) / num_parallel_cells)**2 * single_cell_ir
    
#     # Previous power out of cell calculation
#     # cell_p_out = (air_tc)*(dataDict['Battery Temp'][i]-air_temp) + water_tc*(dataDict['Battery Temp'][i]-water_temp)

#     # Power out of cell to heatsink
#     cell_p_out = (dataDict['Battery Temp'][i] - dataDict['Heatsink Temp'][i]) / (thermal_resistance_SE * num_parallel_cells)

#     # First temp calculations
#     dataDict['Battery Temp'][i+1] = dataDict['Battery Temp'][i] + dt * (dataDict['Dissipated Power'][i] - cell_p_out) / (battery_heat_capacity)

#     # Heatsink temp calculations
#     # Split up calculation:
#     calculation1 = (dataDict['Battery Temp'][i] - dataDict['Heatsink Temp'][i]) * num_series_cells / thermal_resistance_SE - air_tc * (dataDict['Heatsink Temp'][i] - air_temp)
#     dataDict['Heatsink Temp'][i+1] = dataDict['Heatsink Temp'][i] + dt / (heatsink_mass * heatsink_cv) * calculation1

#     ########################################################################
#     # # Determine which file we should reference for voltage plots
#     # # calculate C-rate:
#     # CRate = dataDict['Pack Current'][i] / capacity0

#     # # determine which file we should look in
#     # # options: 0.5, 1, 5, 10
#     # actual_options = [0.5, 1, 5, 10]
#     # CRate_options = [0, 0, 0, max_CRate]
#     # for j in range(0,len(actual_options) - 1):
#     #     CRate_options[j] = (actual_options[j+1] - actual_options[j]) / 2 + actual_options[j]     # iffy algorithm to determine which file to look in
#     #     # basically just finding the average between each of two points

#     # # now determine the location based on the weird thing I made above to deal with the uneven spacing
#     # location = np.searchsorted(CRate_options, CRate)

#     # # Now find the correct file
#     # filename = str(actual_options[location]) + "CDischargeMalloryCSV_out.csv"

#     # # Now let's open the file and read it to a dataframe
#     # df = pd.read_csv(filename, header=[0])
#     # CRateheaders = df.columns.tolist()

#     # # Now read specific rows to lists for purposes that become apparent later
#     # SoC = df.loc[:,CRateheaders[-1]].to_list()
#     # Voltage = df.loc[:,CRateheaders[1]].to_list()

#     # # Now that we have our lists, we look for the location of the SoC algorithm
#     # searchForSoC = SoClookup(SoC, dataDict['SoC Capacity'][i])
#     # SoCIndex = SoC.index(searchForSoC)

#     # # Now we determine pack voltage
#     # dataDict['Pack Voltage'][i+1] = num_series_cells * Voltage[SoCIndex]      # from that index, we then determine the voltage

#     return dataDict         # then we return our favourite dictionary :)

# # State of Charge - energy based
# def SoCenergy(dataDict, totalEnergy):

#     # vector calculation at each point
#     dataDict['SoC Energy'] = (totalEnergy - dataDict['Energy Use']) / totalEnergy * 100

#     return dataDict

# # Plot details to make my life cleaner :))
# def plotDetails(x_axis, y_axis, plotTitle, ax):
#     ax.set_title(plotTitle)
#     ax.set_xlabel(x_axis)
#     ax.set_ylabel(y_axis)
#     ax.grid(True)

#     return

# # Plot the data
# def plotData(dataDict):
#     # Plot the data in this function
#     ROWS = 2
#     COLS = 2
#     FIGWIDTH = 20
#     FIGHEIGHT = 12

#     # create subplots
#     fig, ax = plt.subplots(ROWS, COLS, figsize=(FIGWIDTH, FIGHEIGHT))
#     supTitle = "Point Mass Vehicle Simulation - " + TRACK.replace(".csv","")
#     fig.suptitle(supTitle)

#     # Plot 1)
#     # Regen vs Distance
#     row = 0; col = 0
#     x_axis = "Time (s)"
#     y_axis = "Torque (Nm)"
#     plotTitle = "Motor Torque vs Time"
#     # Convert the velocity from m/s to km/h
#     # dataDict['v0'] = dataDict['v0'] * 3.6
#     ax[row][col].plot(dataDict["t0"][0:700], dataDict["T_m"][0:700])       # plot the data
#     plotDetails(x_axis, y_axis, plotTitle, ax[row][col])  # add the detaila

#     # Plot 2)
#     # Position vs Time
#     row = 0; col = 1
#     x_axis = "Time (s)"
#     y_axis = "SoC(c) (%)"
#     plotTitle = "SoC(c) vs Time"
#     ax[row][col].plot(dataDict["t0"], dataDict["SoC Capacity"])
#     plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

#     # Plot 3)
#     # Battery power vs time with Energy use overlay
#     row = 1; col = 0
#     x_axis = "Distance (m)"
#     y_axis = "Battery Power (kW)"
#     plotTitle = "Battery Power vs Distance over 1 lap"
#     ax[row][col].plot(dataDict["r0"][0:1400], dataDict["P_battery"][0:1400])
#     plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

#     # Plot 4)
#     # SoC energy vs time
#     row = 1; col = 1
#     x_axis = "Time (s)"
#     y_axis = "Temperature (C)"
#     plotTitle = "Battery Temperature vs Time"
#     ax[row][col].plot(dataDict['t0'], dataDict['Battery Temp'])
#     # Will also plot a red line to show the minimum voltage
#     # ax[row][col].plot(dataDict['t0'], np.ones_like(dataDict['t0']) * pack_min_voltage, 'r')
#     plotDetails(x_axis, y_axis, plotTitle, ax[row][col])

#     figTitle = "Point Mass Vehicle Simulation_" + TRACK.replace(".csv","") + ".png"
#     plt.savefig(figTitle)

#     return
