This directory contains source code for running simulations.

Both replicating simulations and steady state simulations call a translocator from the **translocators** subfolder. Translocators run the 1D simulations.

The **confinement volumes** subfolder contains functions that calculate the height of a cell and the positions of tethered origins at a given replication stage. Currently, one file modelled after *C. crescentus* is included; this can in principle be used as a template.
