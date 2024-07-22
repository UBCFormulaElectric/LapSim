import os
import pandas as pd

# # Get current file path
# currentPath = os.path.dirname(__file__)

# # new path
# newPath = os.path.relpath('..\\sim_inputs_and_outputs\\LapSimConstants_Molicel_26A.csv', currentPath)

# df = pd.read_csv(newPath)

filename = "LapSimConstants_Molicel_26A.csv"
path = "C:\VSCode_Python\LapSim_V2\sim_inputs_and_outputs\LapSimConstants_Molicel_P26A.csv"

df = pd.read_csv(path)

# pd.read_csv(r"C:\VSCode_Python\LapSim_V2\sim_inputs_and_outputs\LapSimConstants_Molicel_P26A.csv")



########
# Ok - so now that I caught the missing "2" - I'm gonna need to check the results to see what energy use makes sense and power draw makes sense...
# The power draw for acceleration is WAY TOO LOW now, but I'll check the energy and power for endurance first
# Power draw seems waaaayyyy too low, but maybe now I need to multiply by 4 and then it'll be accurate?