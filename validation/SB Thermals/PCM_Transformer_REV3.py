import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import datetime

# Record date and time for final plot
currentTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

### VALUES TO PLAY WITH!!
# !!!
# h_air = 1                     # W/m2K - change to desired value
h_air = np.linspace(20, 30, 4)  # Range of values for air htc
percent_conduction = 0.3        # Percent of wire mass contacting toroid for conductive heat transfer
T_air = 35                      # C - air ambient temperature
max_wire_T = 120                # C - max acceptable wire temperature
primary_gauge = 14              # AWG - desired wire gauge as calculated in script REV2
secondary_gauge = 10            # AWG - desired wire gauge as calculated in script REV2

# Simulation Time Parameters
dt = 0.1                # s - time step for simulation
total_time = 30 * 60    # s - equivalent to ~30 minutes
total_i = int(total_time / dt)

####### SET UP PARAMETERS:

# Primary coil:
N_prim = 32             # Number of turns - primary coil
I_prim = 4.47           # A - primary coil RMS current

# Secondary coil:
N_sec = 6               # Number of turns - secondary coil
I_sec = 25              # A - secondary coil RMS current

# Copper Wire Material Parameters
k_cu = 398                      # W/mk - thermal conductivity of copper
k_ins = 0.7                     # W/mk - thermal conducivity of wire insulation
c_cu = 385                      # J/kgK - specific heat capacity of copper
#!!!!!!!! CHECK HOW TO COMBINE HEAT CAPACITIES OF THE TWO ELEMENTS!!
rho_m_cu = 8960                 # kg/m3 - mass density of copper

# Transformer - Ferrite Core (fc) - Parameters:
mass_fc = 0.164         # kg - transformer core mass
Ax_fc = 262 / 1000**2   # m2 - cross sectional area
OD_fc = 51.51 / 1000    # m - coated outer diameter
ID_fc = 24 / 1000       # m - coated inner diameter
HT_fc = 21.59 / 1000    # m - coated height
As_fc = math.pi * HT_fc * (OD_fc + ID_fc) + math.pi/2 * (OD_fc**2 - ID_fc**2)
c_fc = 800              # J/kgK - specific heat of ferrite core
k_fc = 3.5              # W/mK - thermal conductivity

wrap_perimeter = (OD_fc - ID_fc) + 2 * HT_fc    # m - toroid wire wrap perimeter for one turn
l_fc = (OD_fc - ID_fc) / 2                      # m - Assumed "length" of the core for thermal resistance calculations
R_th_fc = math.log(OD_fc / ID_fc) / (k_fc * 2 * math.pi * HT_fc)                 # K/W - thermal resistance of ferrite core - eqn 3.28

# Electrical Parameters
f_sw = 148 * 10**3              # Hz - maximum switching frequency
mu_cu = 1                       # Relative permeability of copper
rho_R_cu = 1.68 * 10**-8        # Ohm-m - resitivity of copper
mu_vac = 4 * math.pi * 10**-7   # H/m - permeability of a vacuum
s_cu = math.sqrt(rho_R_cu / (math.pi * f_sw * mu_cu * mu_vac))   # m - skin depth of copper

# Wire Parameters and class setup
class wire_params:
    # Class Item Definition
    def __init__(self, wire_gauge, wire_data, I, N): ### !!! TEMPORARILY ADDED A "j" value
        # Calculate remaining wire parameters
        self.I = I                                              # A - expected current through wire
        self.N = N                                              # turns - number of required turns

        # had issues the first time i did this bc i forgot np searchsorted only works with ascending arrays

        index = np.searchsorted(-wire_data.loc[:,'Gauge'], -wire_gauge)           # Index for further searches
        self.D_cu = wire_data.loc[index, 'Conductor Diameter'] / 1000                  # m - Copper conductor diameter
        self.D_ins = wire_data.loc[index, 'Finished Wire Diameter'] / 1000             # m - total wire diameter including insulation
        self.TH_ins = wire_data.loc[index, 'Enameled Insulation Thickness'] / 1000     # m - insulation thickness radially
        self.gauge = wire_gauge                                                 # AWG - wire gauge

        self.length = N * wrap_perimeter                        # m - total wire length
        self.Ax = math.pi / 4 * self.D_cu**2                    # m2 - cross sectional area of copper conductor
        self.mass = rho_m_cu * self.length * self.Ax            # kg - mass of copper conductor
        self.AxR = math.pi / 4 * (self.D_cu**2 - (self.D_cu - s_cu)**2)   # m2 - cross sectional area for electrical resistance calcc
        self.R = rho_R_cu * self.length / self.AxR              # Ohms - electrical resistance of copper conductor
        self.total_As = math.pi * self.D_ins * self.length      # m2 - total surface area
        self.cond_As = self.total_As * percent_conduction       # m2 - surface area for conduction
        self.conv_As = self.total_As * (1 - percent_conduction) # m2 - surface area for convection
        self.R_th = math.log(self.D_ins / self.D_cu) / (k_ins * self.cond_As)        # K/W - thermal resistance for conduction - Eqn 3.28

####### UPLOAD AND ANALYZE WIRE DATA
wire_data = pd.read_csv("Wire_Gauge_Data.csv", header=[0], skiprows=[1])    # skip the row with the units since I don't want to deal with that
wire_data = wire_data.iloc[::-1,:].reset_index(drop=True)                   # reverse rows and reset index

###### CHOOSE THE DESIRED WIRE!
primary = wire_params(primary_gauge, wire_data, I_prim, N_prim)
secondary = wire_params(secondary_gauge, wire_data, I_sec, N_sec)

##### PRINT WIRE PARAMETERS FOR DEBUGGING PURPOSES

attrs = vars(primary)
print("PRIMARY:")
print(', '.join("%s: %s" % item for item in attrs.items()))

attrs = vars(secondary)
print("SECONDARY:")
print(', '.join("%s: %s" % item for item in attrs.items()))

####### CONDUCT THERMAL SIMULATION OVER TIME

# Temperature lists for primary, secondary, and core

# Lists for all data
T_1_data = []
T_2_data = []
T_C_data = []

for j in range(0, len(h_air)):
    # Reset time variable
    time = [0]                  # s
    T_1 = np.array([T_air])     # K
    T_2 = np.array([T_air])     # K
    T_C = np.array([T_air])     # K
    Qgen = np.array([0])        # W

    for i in range(0, total_i):

        T_coil_coeff = lambda coil : coil.mass * c_cu / dt + 1 / (coil.R_th + R_th_fc) + h_air[j] * coil.conv_As
        T_core_coeff = lambda coil : -1 / (coil.R_th + R_th_fc)
        T_constant = lambda coil, T0: coil.I**2 * coil.R + coil.mass * c_cu / dt * T0 + h_air[j] * coil.conv_As * T_air

        # Set up equation 1:
        T_1_coeff = T_coil_coeff(primary)
        T_1_core = T_core_coeff(primary)
        T_1_constant = T_constant(primary, T_1[i])

        # Set up for equation 2:
        T_2_coeff = T_coil_coeff(secondary)
        T_2_core = T_core_coeff(secondary)
        T_2_constant  = T_constant(secondary, T_2[i])

        # Set up for equation 3:
        T_C_coeff = 1 / (primary.R_th + R_th_fc) + 1 / (secondary.R_th + R_th_fc) + mass_fc * c_fc / dt + h_air[j] * As_fc
        T_C_1_coeff = -1 / (primary.R_th + R_th_fc)
        T_C_2_coeff = -1 / (secondary.R_th + R_th_fc)
        T_C_constant = mass_fc * c_fc / dt * T_C[i] + h_air[j] * As_fc * T_air

        # Create matrix to solve
        solving_matrix = np.array([[T_1_coeff, np.zeros_like(T_1_coeff), T_1_core],
                                [np.zeros_like(T_1_coeff), T_2_coeff, T_2_core],
                                [T_C_1_coeff, T_C_2_coeff, T_C_coeff]])
        solving_vector = np.array([[T_1_constant],
                                [T_2_constant],
                                [T_C_constant]])

        # Solve:
        result = np.linalg.solve(solving_matrix, solving_vector)

        # Save results to new temperature arrays
        T_1 = np.append(T_1, result[0])
        T_2 = np.append(T_2, result[1])
        T_C = np.append(T_C, result[2])
        time.append(time[i] + dt)
        Qgen = np.append(Qgen, primary.I**2 * primary.R + secondary.I**2 + secondary.R)

    # Add to full data
    T_1_data.append(T_1)
    T_2_data.append(T_2)
    T_C_data.append(T_C)

###### VISUALIZE RESULTS
time = np.array(time)

# Make subplots of each:
ROWS = 2
COLS = 2
FIGWIDTH = 10
FIGHEIGHT = 5
fig, ax = plt.subplots(ROWS, COLS, figsize=(FIGWIDTH, FIGHEIGHT))
supTitle = "Temperature Sim for PCM Transformer"
fig.suptitle(supTitle)

# Plot for Primary Coil
row = 0
col = 0

# Plot four different cases
for i in range(0, len(h_air)):
    T_1 = T_1_data[:][i] # might need to switch indices
    T_2 = T_2_data[:][i]
    T_C = T_C_data[:][i]

    ax[row][col].plot(time, T_1, label='Primary')
    ax[row][col].plot(time, T_2, label="Secondary")
    ax[row][col].plot(time, T_C, label="Ferrite Core")
    ax[row][col].legend()
    ax[row][col].set_xlabel("Time (s)")
    ax[row][col].set_ylabel("Temperature (C)")
    ax[row][col].set_title("PCM Transformer Thermals with htc: %.1f W/m2K" % h_air[i])
    ax[row][col].grid(True)

    # Move to the next subplot location
    if col == 0:
        col = col + 1
    else:
        col = 0
        row = row + 1

plt.tight_layout(pad=1)

# # Make plot of results:
# plt.plot(time, T_1, label='Primary')
# plt.plot(time, T_2, label="Secondary")
# plt.plot(time, T_C, label="Ferrite Core")
# plt.legend()
# plt.grid(True)
# plt.xlabel("Time (s)")
# plt.ylabel("Temperature (C)")
# plt.title("PCM Transformer Thermals with %i gauge Primary and %i gauge Secondary" % (primary_gauge, secondary_gauge))

fig_path = "plots/" + currentTime + "_Resulting_Wire_Temperature"+ ".png"
plt.savefig(fig_path)
plt.show()