import pickle
import os
import time
import numpy as np
import polychrom
from polychrom import polymerutils
from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file
from polychrom.polymer_analyses import generate_bins
from polychrom.polymerutils import load_URI,  load
from polychrom import contactmaps
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file
import openmm 
import shutil
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},reload_support=True)
import warnings
import h5py as hp
import glob
from itertools import product
import re

from ..confinement_volumes.c_crescentus import *

def run_simulation(monomer_size,monomer_wig, N,\
        steps_per_sample, save_folder, saveEveryConfigs, \
        savedSamples, restartConfigurationEvery, GPU_choice = 0,\
        col_rate=0.1, trunc=0.5):
    """
    This function runs simulations of a linear, unconfined polymer. 

    monomer_size : float
    The size of the monomer in nanometers.

    monomer_wig : float
    The monomer wiggle distance in simulation units.
    
    steps_per_sample : int
    The number of  polymer simulation integration steps to be carried out per block.

    save_folder : str
    Location where to dump data

    saveEveryConfigs : int
    Frequency of saving data.

    savedSamples : int
    Total number of blocks to save.

    restartConfigurationEvery : int
    Frequency of restarting the pre-allocation of bonds.

    GPU_choice : int
    The GPU number on which to run the simulation.

    col_rate : float
    Sets the friction for the polymer simulations

    trunc : float
    The excluded volume strength.

    """

    # assertions for easy managing code below 
    assert restartConfigurationEvery % saveEveryConfigs == 0 
    assert (savedSamples * saveEveryConfigs) % restartConfigurationEvery == 0 

    savesPerInit = restartConfigurationEvery // saveEveryConfigs
    # InitsSkip = saveEveryConfigs * skipSamples  // restartConfigurationEvery
    InitsSave = savedSamples * saveEveryConfigs // restartConfigurationEvery
    InitsTotal  = InitsSave #+ InitsSkip
    print("Polymer will be initialized {0} times.".format(InitsTotal))    

    block=0 #start count from zero   
    # clean up the simulation directory
    folder = save_folder
    if os.path.exists(folder):
        shutil.rmtree(folder)

    # create the starting conformation
    start_data = start_point_linear_free(N) #random walk of length N, with step size = 1 simulation unit

    # create the reporter class
    reporter = HDF5Reporter(folder=folder, max_data_length=100)

    # Iterate over various BondUpdaterInitializations
    for BondUpdaterCount in range(InitsTotal):
        # simulation parameters are defined below 
        a = Simulation(
                platform="cuda",
                integrator="variableLangevin", 
                error_tol=0.001,
                GPU = "{}".format(GPU_choice), 
                collision_rate=col_rate, 
                N = len(start_data),
                max_Ek=100.,
                reporters=[reporter],
                PBCbox=False, #EDIT: no boundaries
                precision="mixed",
                verbose=True)  # timestep not necessary for variableLangevin

        a.set_data(start_data)  # loads polymer.

        # -----------Adding forces ---------------
        a.add_force(
            forcekits.polymer_chains(
                a,
                chains=[(0, N, False)], #LINEAR chain
                bond_force_func=forces.harmonic_bonds, # adds harmonic bonds for polymers
                bond_force_kwargs={
                    'bondLength':1.,
                    'bondWiggleDistance':monomer_wig, # Bond distance will fluctuate this much
                },

                angle_force_func=forces.angle_force,
                angle_force_kwargs={
                    'k':0.05 # we are making a very flexible polymer, basically not necessary here
                },
                nonbonded_force_func=forces.polynomial_repulsive, # this is the excluded volume potential
                nonbonded_force_kwargs={
                    #'trunc': trunc, # this will let chains cross sometimes (energy in kb*T units)
                    'trunc': trunc,
                    'radiusMult':1.05, # this is from old code
                    #'trunc':10.0, # this will resolve chain crossings and will not let chain cross anymore
                },
                except_bonds=True,
            )
        )

        print("\nNot adding tethers or confinement")
        a.step = block

        # Minimize energy for first bonds
        print("Polymer burn-in")
        a.local_energy_minimization() 
        #Docs say first steps after energy minimization have large error; don't save
        a.integrator.step(100*steps_per_sample)

        a.step=block
        step_saved=0
        # Iterate over simulation time steps within each BondUpdater 
        for i in range(restartConfigurationEvery-2):
            # BondUpdater updates bonds at each step.
            if i % saveEveryConfigs == 0: #save just before resetting configuration  
                # save monomer positions
                step_saved+=1
                a.do_block(steps=steps_per_sample, save_extras={"time_step":step_saved}) 
            else:
                a.integrator.step(steps_per_sample)  #do steps without getting the positions from the GPU (faster)

        block = a.step
        del a

        time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)

    # dump data to output file
    reporter.dump_data()
    done_file = open(os.path.join(folder,'sim_done.txt'),"w+")
    done_file.close()
    del reporter

