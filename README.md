# PINN_Geothermal3D
**This program is a physics-informed neural network (PINN) for modeling temperatures, pressures, and permeabilities of natural-state geothermal systems.**
![Image](https://github.com/user-attachments/assets/1b6bbf22-e133-4059-a99a-73e61af19d56)
In geothermal areas, temperature, pressure and permeability structures are generally modeled from the physical quantities obtained at available wells, and the constructed model is used for evaluating geothermal resource potential, selecting drilling locations, and planning geothermal energy production. We have proposed a method that uses PINNs to model temperatures, pressures, and permeabilities from well data.

PINNs conducts network training by considering physical laws and boundary conditions in addition to the data. This program assumes a subsurface porous medium saturated with pure water as geothermal systems, and considers the mass and energy balance equations for a single-phase liquid in a steady state condition as physical laws.
## Getting started
This program is written based on TensorFlow on a jupyter-notebook platform, and was developed with the following modules:
```
Tensorflow version 2.9.1
Tensorflow probability version 0.17.0
Numpy version 1.21.1
pandas version 1.4.3
matplotlib version 3.5.0
scipy version 1.4.1
```
To run PINN_Geothermal_naturalstate.ipynb, put the program together with utils and optimizer directories as well as csv datasets (Reference_model.csv and Welldata_30wells.csv) in the same directory, and execute the jupyter-notebook program.
## Outputs
<ins>Outputs are three-dimensional temperatures, presures, logarithm of permeabilities at the target domain.</ins>
The 3D coordinates (X, Y, Elevation) of the target domain are read from a CSV file.
The "save_predicts" directory is created at the beginning of the program, and the csv files with temperatures, pressures, logarithm of permeabilities are output for each specified epoch.
The "save_checkpoints" directory is also created by the program, and The checkpoint files and loss histories are output to this directory.
The loss history csv file contains not only the loss for each epoch of the training and validation data, but also the loss components (Temperature_loss, Pressure_loss, Permeability_loss, Mass_balance_loss, Energy_balance_loss etc).

## Citation
Kazuya Ishitsuka, Keiichi Ishizu, Norihiro Watanabe, Yusuke Yamaya, Anna Suzuki, Toshiyuki Bandai, Yusuke Ohta, Toru Mogi, Hiroshi Asanuma, Tatsuya Kajiwara, & Takeshi Sugimoto

"Toward reliable and practical inverse modeling of natural-state geothermal systems using physics-informed neural networks: three-dimensional model construction and assimilation with magnetotelluric data", Under Review.
