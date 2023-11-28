# Loop-extruders alter bacterial chromosome topology to direct entropic forces for segregation

This repository contains the code used to simulate replicating bacterial chromosomes inside a confinement. The code has been adapted from https://github.com/hbbrandao/bacterialSMCtrajectories. The main changes are the inclusion of replication in both the 1D and 3D simulations, as well as the addition of a growing cylindrical confinement.

Analysis of the data was done using the Julia code in the repository: https://github.com/PLSysGitHub/loop-extrusion_with_replication_analysis

The repository includes two scripts that can be run from the command line with a number of options. replicating_sim.py can be run to generate configurations with ongoing replication. steady_state_sim.py can be run to sample configurations from simulations with fixed replication fork positions. Call python replicating_sim.py --help for descriptions of the command line options.

## Code dependencies: 
https://github.com/open2c/polychrom
Please replace the polychrom/polychrom/forces.py file with the version included in this repository

By default, the code runs using CUDA, which can be installed at https://developer.nvidia.com/cuda-downloads. If you do not have a CUDA-compatible GPU, you can edit the option platform="cuda" to platform="CPU" in the a=simulation(...) call of the simulation you want to run.

## References:
Brandão, H. B., Ren, Z., Karaboja, X., Mirny, L. A., & Wang, X. (2021). DNA-loop extruding SMC complexes can traverse one another in vivo. *Nat. Struct. Mol. Biol.* 

## Questions?
If you have any questions, feel free to contact j.k.harju[at]vu.nl
