# Lap Sim Version 2
### Mallory Moxham, July 2024

This is the most recent version of the lap simulation. I am now keeping it up-to-date on github. Yayyyyyyyy!!!

## How do I use this?
* Please download ALL files.
* Please put ALL download files into a parent folder called: "LapSim_V2" - this means that you won't have a to change any file paths!
* I use VSCode to run this.
* Packages:
    * **OPTION 1:** I use Python 3.12... (most recent), pandas, numpy, matplotlib, csv, json (pip install *library*)
    * **OPTION 2:** You can also use Anaconda and upload the .yml file that I've added in the main folder. This has all of the packages that I use for this simulation. But there's nothing really fancy here, so there's no need to use the virtual env if you don't want to.
        * Please see this website for details on how to add the environment to your computer: https://docs.anaconda.com/working-with-conda/environments/#creating-an-environment-from-an-environment-yml-file 
* To change constants, please use the file: "LapSimConstants.csv" in "sim_inputs_and_outputs" folder
    * At the moment, please do *not* set regen_on = TRUE or motor_choice = emrax
    * The constants I suggest you change are:
        * track_choice
        * cell_choice
        * torque_speed
* To see results at end, please view graphs in sim_inputs_and_outputs folder under the specific cell type you've chosen. Also see dynamicsCalcs.csv for a full output of variables.
    * At the moment, there are a few variables which are not being analyzed. These may show up with zeros... just ignore them! There's also a description of what the variables are and their units in the lapsim_V2.0.py script.
* You can leave the LapSimConstants.csv file open while running the program, but please close the dynamicsCalcs.csv file or else the program will not have permission to update it and will throw an error.

## Notes on Changes and Action Items
### Changes From Previous Model
* This model now has braking involved.
* It includes the data from the AMK motors, including:
    * New gear ratio
    * Current/RPM/Torque data from manufacturer
* Now iterates with constant distance interval instead of time. This will give more accurate time estimates.
* Now includes effects of aero package (ie. downforce and drag).
* Includes the voltage sag with different current usage.

### Notes on Current Issues and Next Steps
* Please note that this is a POINT MASS SIMULATION! It does not involve specific suspension/vehicle dynamics analysis.
* Some constants are not verified, namely:
    * longitudinal and lateral friction coefficients
    * Any constants covering battery thermals
    * Drag and lift coefficients
    * Rolling resistance coefficient
    * Maximum centripetal acceleration (I used this for the braking data)
    * Car weight is inaccurate - as it is using melasta and emrax information still
* Battery thermal model will also need to be redone as we will be using new cells with different IR and thermal characteristics
* Still need to add analysis for REGEN - shouldn't be super difficult based on this new set-up

## Validation Files
To get the correct files for the validation testing, please see this google drive folder: https://drive.google.com/open?id=1DonZTSyBmcpJp6LTFso0GhIdvSxRCfn9&usp=drive_fs 

## Questions?
Please reach out to me if you want help with navigating this! I am happy to show you over a call or in-person how to work with the sim. Also let me know if you want specific features or code added to it, and I'd be happy to work with you to integrate your ideas.