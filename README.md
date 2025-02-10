# PINN_Geothermal3D
**This program is a physics-informed neural network (PINN) for modeling temperatures, pressures, and permeabilities of natural-state geothermal systems.**

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
## Citation
Kazuya Ishitsuka, Keiichi Ishizu, Norihiro Watanabe, Yusuke Yamaya, Anna Suzuki, Toshiyuki Bandai, Yusuke Ohta, Toru Mogi, Hiroshi Asanuma, Tatsuya Kajiwara, & Takeshi Sugimoto, Toward reliable and practical inverse modeling of natural-state geothermal systems using physics-informed neural networks: three-dimensional model construction and assimilation with magnetotelluric data, Under Review.
