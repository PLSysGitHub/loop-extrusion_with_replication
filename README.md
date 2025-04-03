# Loop-extrusion simulations on bacterial chromosomes.

This repository contains code used to simulate replicating bacterial chromosomes inside a confinement. The code has been adapted from https://github.com/hbbrandao/bacterialSMCtrajectories. The main changes are the inclusion of replication in both the 1D and 3D simulations, as well as the addition of a growing cylindrical confinement.

This new version uses an adapted version of Anton Goloborodko's looplib package for loop-extruder simulations. Adapted version considers a circular chromosome, and has been expanded to do simulations on a replicating chromosome. Note that the 3D simulations just take files with loop-extruder trajectories as an input, meaning that different loop-extruder simulations could easily be used.

The main scripts can be used to simulate either a replicating system (replicating...py) or a non-replicating system (steady_state...py). There are separate scripts for simulations with or without loop-extruders.

Additionally, there are scripts where the spring lengths and stiffnesses are dynamically adjusted behind and ahead of the replication forks, as discussed in our manuscript [Learning the dynamic organization of a replicating bacterial chromosome from time-course Hi-C data](FIXME).

## Code dependencies: 
https://github.com/open2c/polychrom
Please replace the polychrom/polychrom/forces.py file with the version included in this repository. This file contains a new confinement potential; a cylinder with rounded caps, as well as excluded volume potentials where the strength can be set to zero, necessary for replicating simulations.

By default, the code runs using CUDA, which can be installed at https://developer.nvidia.com/cuda-downloads. If you do not have a CUDA-compatible GPU, you can edit the option platform="cuda" to platform="CPU" in the a=simulation(...) call of the simulation you want to run.

## Run times:
The default steady state simulations with loop-extrusion took a day or two to run on a NVIDIA GeForce RTX 3080 GPU. The default replicating simulations with loop-extrusion took approximately a day. Simulations without loop-extruders are significantly faster, and typically finish in less than a day.

## References:
Brandão, H. B., Ren, Z., Karaboja, X., Mirny, L. A., & Wang, X. (2021). DNA-loop extruding SMC complexes can traverse one another in vivo. *Nat. Struct. Mol. Biol.* 

## Questions?
If you have any questions, feel free to contact j.k.harju[at]vu.nl
