import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

### VALUES TO PLAY WITH!!
# !!!
h_air = 1                       # W/m2K - change to desired value
percent_conduction = 0.5        # Percent of wire mass contacting toroid for conductive heat transfer
T_air = 35                      # K - air ambient temperature
# p = 38                          # For manually iterating through SIZE OPTIONS

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

# # !!! DEBUGGING - to see if temperature increases with NO heat transfer out of the wires
# k_cu = 10**-15
# k_ins = 10**-15
# k_fc = 10**-15

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
    def __init__(self, wire_type, I, N, j): ### !!! TEMPORARILY ADDED A "j" value
        # Calculate remaining wire parameters
        self.I = I                                              # A - expected current through wire
        self.N = N                                              # turns - number of required turns
        self.AWG = wire_type['Gauge'][j]
        self.D_cu = wire_type['D_cu'][j]                        # m - copper conductor diameter
        self.D_ins = wire_type['D_ins'][j]                      # m - total wire diameter including insulation
        self.Th_ins = wire_type['Th_ins'][j]                    # m - insulation thickness
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

# Coil Parameters - just conductor and insulated conductor
D_cu = wire_data.loc[:,'Conductor Diameter'].to_numpy()
D_ins = wire_data.loc[:,'Finished Wire Diameter'].to_numpy()
Th_ins = wire_data.loc[:,'Enameled Insulation Thickness'].to_numpy()
Gauge = wire_data.loc[:, 'Gauge'].to_numpy()

# Set up empty arrays for all values:
    # D_cu = copper conductor diameter
    # D_ins = total wire diameter including insulation
    # Th_ins = insulation thickness

D_cu_1 = []
D_cu_2 = []
D_ins_1 = []
D_ins_2 = []
Th_ins_1 = []
Th_ins_2 = []
Gauge_1 = []
Gauge_2 = []

# Create combination arrays
num = len(D_cu)
for i in range(0, num):         # For the primary coil

    for j in range(i + 1, num): # For all larger values than the primary in the array
        D_cu_1.append(D_cu[i])
        D_cu_2.append(D_cu[j])

        D_ins_1.append(D_ins[i])
        D_ins_2.append(D_ins[j])

        Th_ins_1.append(Th_ins[i])
        Th_ins_2.append(Th_ins[j])

        Gauge_1.append(Gauge[i])
        Gauge_2.append(Gauge[j])

# Convert all outputs to numpy and convert from mm to m to keep units correct
D_cu_1 = np.array(D_cu_1) / 1000
D_cu_2 = np.array(D_cu_2) / 1000
D_ins_1 = np.array(D_ins_1) / 1000
D_ins_2 = np.array(D_ins_2) / 1000
Th_ins_1 = np.array(Th_ins_1) / 1000
Th_ins_2 = np.array(Th_ins_2) / 1000
Gauge_1 = np.array(Gauge_1)
Gauge_2 = np.array(Gauge_2)

# Set up dictionaries with primary and secondary coil information
temp_1 = [D_cu_1, D_ins_1, Th_ins_1, Gauge_1]
temp_2 = [D_cu_2, D_ins_2, Th_ins_2, Gauge_2]
keys = ['D_cu', 'D_ins', 'Th_ins', 'Gauge']

coil_prim = dict(zip(keys, temp_1))
coil_sec = dict(zip(keys, temp_2))


###### CHOOSE THE DESIRED WIRE!
# Lowest temperatures
lowest_T_1 = 10**10
lowest_T_2 = lowest_T_1
lowest_T_C = lowest_T_1

# Lists for all data
T_1_data = []
T_2_data = []
T_C_data = []

for p in range(0, len(D_cu_1)):

    primary = wire_params(coil_prim, I_prim, N_prim, p)
    secondary = wire_params(coil_sec, I_sec, N_sec, p)

    ##### PRINT WIRE PARAMETERS FOR DEBUGGING PURPOSES

    # attrs = vars(primary)
    # print("PRIMARY:")
    # print(', '.join("%s: %s" % item for item in attrs.items()))

    # attrs = vars(secondary)
    # print("SECONDARY:")
    # print(', '.join("%s: %s" % item for item in attrs.items()))

    ####### CONDUCT THERMAL SIMULATION OVER TIME
    dt = 0.1                # s - time step for simulation
    total_time = 30 * 60    # s - equivalent to ~30 minutes
    total_i = int(total_time / dt)

    # Temperature lists for primary, secondary, and core
    T_1 = np.array([T_air])     # K
    T_2 = np.array([T_air])     # K
    T_C = np.array([T_air])     # K
    Qgen = np.array([0])        # W
    time = [0]                  # s

    for i in range(0, total_i):
        # Add for loop later
        T_coil_coeff = lambda coil : coil.mass * c_cu / dt + 1 / (coil.R_th + R_th_fc) + h_air * coil.conv_As
        T_core_coeff = lambda coil : -1 / (coil.R_th + R_th_fc)
        T_constant = lambda coil, T0: coil.I**2 * coil.R + coil.mass * c_cu / dt * T0 + h_air * coil.conv_As * T_air

        # Set up equation 1:
        T_1_coeff = T_coil_coeff(primary)
        T_1_core = T_core_coeff(primary)
        T_1_constant = T_constant(primary, T_1[i])

        # Set up for equation 2:
        T_2_coeff = T_coil_coeff(secondary)
        T_2_core = T_core_coeff(secondary)
        T_2_constant  = T_constant(secondary, T_2[i])

        # Set up for equation 3:
        T_C_coeff = 1 / (primary.R_th + R_th_fc) + 1 / (secondary.R_th + R_th_fc) + mass_fc * c_fc / dt + h_air * As_fc
        T_C_1_coeff = -1 / (primary.R_th + R_th_fc)
        T_C_2_coeff = -1 / (secondary.R_th + R_th_fc)
        T_C_constant = mass_fc * c_fc / dt * T_C[i] + h_air * As_fc * T_air

        # Create matrix to solve
        solving_matrix = np.array([[T_1_coeff, 0, T_1_core],
                                [0, T_2_coeff, T_2_core],
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

    ###### WHICH SOLUTION PERFORMS THE BEST FOR PRIMARY AND SECONDARY?
    T_1_max = max(T_1)
    T_2_max = max(T_2)
    T_C_max = max(T_C)

    # Change to stay underneath an acceptable limit (ie. 150 C)
    # This allows us to be less conservative with primary winding estimates
    if T_1_max < 150 and lowest_T_1 > 150:
        lowest_T_1 = T_1_max
        best_combo_1 = p
    
    # This must still stay this way
    if T_2_max < lowest_T_2:
        lowest_T_2 = T_2_max
        best_combo_2 = p

    # # Core temperature is not problematic, can comment out.
    # if T_C_max < lowest_T_C:
    #     lowest_T_C = T_C_max
    #     best_combo_C = p
    
    T_1_data.append(T_1)
    T_2_data.append(T_2)
    T_C_data.append(T_C)

###### VISUALIZE RESULTS
time = np.array(time)

# For the best results, see:
print("Best Case for Primary Coil: %i with Wire Gauge: %s" % (best_combo_1, Gauge_1[best_combo_1]))
print("Best Case for Secondary Coil: %i with Wire Gauge %s" % (best_combo_2, Gauge_2[best_combo_2]))

# Make subplots of each:
ROWS = 1
COLS = 2
FIGWIDTH = 10
FIGHEIGHT = 5
fig, ax = plt.subplots(ROWS, COLS, figsize=(FIGWIDTH, FIGHEIGHT))
supTitle = "Temperature Sim for PCM Transformer"
fig.suptitle(supTitle)

# Plot for Primary Coil
row = 0
col = 0

T_1 = T_1_data[best_combo_1][:] # might need to switch indices
T_2 = T_2_data[best_combo_1][:]
T_C = T_C_data[best_combo_1][:]

ax[col].plot(time, T_1, label='Primary')
ax[col].plot(time, T_2, label="Secondary")
ax[col].plot(time, T_C, label="Ferrite Core")
ax[col].legend()
ax[col].set_xlabel("Time (s)")
ax[col].set_ylabel("Temperature (C)")
ax[col].set_title("Best Scenario for Primary Winding, Wire Gauge %s" % Gauge_1[best_combo_1])
ax[col].grid(True)

# Plot for Secondary Coil
row = 0
col = 1

T_1 = T_1_data[best_combo_2][:] # might need to switch indices
T_2 = T_2_data[best_combo_2][:]
T_C = T_C_data[best_combo_2][:]

ax[col].plot(time, T_1, label='Primary')
ax[col].plot(time, T_2, label="Secondary")
ax[col].plot(time, T_C, label="Ferrite Core")
ax[col].legend()
ax[col].set_xlabel("Time (s)")
ax[col].set_ylabel("Temperature (C)")
ax[col].set_title("Best Scenario for Secondary Winding, Wire Gauge %s" % Gauge_2[best_combo_2])
ax[col].grid(True)

plt.tight_layout(pad=1)

plt.savefig("Best_Case_Wire_Parameters_2.png")