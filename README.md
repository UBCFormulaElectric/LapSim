# Lap Sim Version 2
### Mallory Moxham, July 2024

This is the most recent version of the lap simulation. I am now keeping it up-to-date on github. I don't have access to my github account at the moment - for 2FA reasons... Once I get back to Canada, I'll put this into the main branch.

## How do I use this?
* Please download ALL files.
* I use VSCode to run this.
* Packages:
    * I use Python 3.12... (most recent)
    * Install the Jupyter notebook extension (can do this through VSCode)
    * Install pandas, numpy, matplotlib
* To change constants, please use the file: "LapSimConstants.csv"
    * At the moment, please do *not* set regen_on = TRUE
* To see results at end, please view graphs in Jupyter notebook OR see dynamicsCalcs.csv for a full output of variables. Jupyter notebook will also show some basic outputs in the terminal.
* You can leave the LapSimConstants.csv file open while running the program, but please close the dynamicsCalcs.csv file or else the program will not have permission to update it and will throw an error.

## Notes on Changes and Action Items
### Changes From Previous Model
* This model now has braking involved
* It includes the data from the AMK motors, including:
    * New gear ratio
    * Current/RPM/Torque data from manufacturer
* Now iterates with constant distance interval instead of time. This will give more accurate time estimates
* Now includes effects of aero package (ie. downforce and drag)

### Notes on Current Issues and Next Steps
* Please note that this is a POINT MASS SIMULATION! It does not involve specific suspension/vehicle dynamics analysis.
* Energy consumption is unreasonably high - around 17 kWh when it should be around 7 kWh. Will need to debug and figure out how to fix this problem...
* Some constants are not verified, namely:
    * longitudinal and lateral friction coefficients
    * Any constants covering battery thermals
    * Drag and lift coefficients
    * Rolling resistance coefficient
    * Maximum centripetal acceleration (I used this for the braking data)
    * Car weight is inaccurate - as it is using melasta and emrax information still
* Currently using Melasta cell data - need to update this
* Battery thermal model will also need to be redone as we will be using new cells with different IR and thermal characteristics
* Still need to add analysis for REGEN - shouldn't be super difficult based on this new set-up
* Add VOLTAGE-SAG (ie. the voltage drop when power is lost to battery IR) - will also want to include voltage/SoC changes in the model too (but I need some more data to do this)