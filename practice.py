import json
import pandas as pd

# Convert .json to dictionary
in_json = 'emrax_data.json'
in_json = open(in_json)
in_json = in_json.read()
emrax_dict = json.loads(in_json)

# Split up dictionary into individual variables
emrax_dict['Motor Efficiency'] = pd.DataFrame(emrax_dict['Motor Efficiency'])

# NOTE - the above might not actually be necessary

# Additional emrax data
phases = 3
pole_pairs = 10
pm = emrax_dict['lambda pm']

# Function handle for managing torque-->current changes if necessary
torque_current = lambda I: phases / 2 * pole_pairs * pm * I

########
# Import AMK Data
in_json = 'AMK_data.json'
in_json = open(in_json)
in_json = in_json.read()
AMK_dict = json.loads(in_json)

# Split dictionary into individual variables
AMK_dict['S']