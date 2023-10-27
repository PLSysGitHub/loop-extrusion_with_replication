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

def get_kb_from_theory(num_extruders):
    d = 358 # kb
    v3 = 0.46
    v2 = 0.59
    v1 = 0.83
    n_encounters = 850*(0.4*num_extruders/1371) 
    tb = d/n_encounters/v3 - d/n_encounters/v2*(1-n_encounters/d)
    kb = 1/tb
    return kb

def run_simulation(R,monomer_size,monomer_wig, N,\
        steps_per_sample, radius,\
        smcBondDist, smcBondWiggleDist, save_folder, saveEveryConfigs, \
        savedSamples, restartConfigurationEvery, GPU_choice = 0, F_z=0.,\
        col_rate=0.1, trunc=0.5, num_tethers=2, start_segregated=False,top_monomer=0,\
        forks_bound=False):
    """
    This function initiates the bacterial chromosome. 

    R : int
    The number of replicated monomers in the polymer.

    monomer_size : float
    The size of the monomer in nanometers.

    monomer_wig : float
    The monomer wiggle distance in simulation units.
    
    steps_per_sample : int
    The number of  polymer simulation integration steps to be carried out per block.

    radius : float
    The radius of the confinement volume in simulation units.

    smcBondDist : float
    Distance of SMC bonds in monomer units

    smcBondWiggleDist : float
    The harmonic bond strength for SMC-type bonds

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

    F_z : float
    The force in the z-direction exerted on particles that are tethered (ie oris)

    col_rate : float
    Sets the friction for the polymer simulations

    trunc : float
    The excluded volume strength.

    num_tethers : int
    The number of tethered particles (ie oris) between 0 and 2.

    start_segregated : bool
    Whether to start the polymer in a segregated state.

    top_monomer : int
    The index of the monomer that will be initialised at the pole.
    """

    # assertions for easy managing code below 
    assert restartConfigurationEvery % saveEveryConfigs == 0 
    assert (savedSamples * saveEveryConfigs) % restartConfigurationEvery == 0 

    savesPerInit = restartConfigurationEvery // saveEveryConfigs
    # InitsSkip = saveEveryConfigs * skipSamples  // restartConfigurationEvery
    InitsSave = savedSamples * saveEveryConfigs // restartConfigurationEvery
    InitsTotal  = InitsSave #+ InitsSkip
    print("Polymer will be initialized {0} times.".format(InitsTotal))    

    pulled=N
    block=0 #start count from zero   
    height=R_to_height(R,N, monomer_size)
    pos_old, pos_new=R_to_z_oris(R,N, height, monomer_size)
    # clean up the simulation directory
    folder = save_folder
    if os.path.exists(folder):
        shutil.rmtree(folder)

    # create the starting conformation
    if start_segregated:
        start_data = start_point_segregated(R,N,height)
    else:
        start_data = start_point_unsegregated(top_monomer,R,N,height)

    #the bonds that will be used, given that unreplicated bonds need to be two times softer
    if forks_bound:
        num_bonds=2*N+(N-R)+1
    else:
        num_bonds=2*N+(N-R)

    wiggle_array=np.ones(num_bonds)*monomer_wig #polymer bonds plus replication bonds
    wiggle_array[R//2:N-R//2]*=np.sqrt(2) #unreplicated polymer bonds
    wiggle_array[N+R//2:2*N-R//2]*=np.sqrt(2) #unreplicated polymer bonds on second copy
    wiggle_array[2*N:-1]=smcBondWiggleDist #replication bonds
    
    length_array=np.ones(num_bonds)
    length_array[2*N:-1]=smcBondDist #replication bonds have this length, mostly for compatibility with dynamic simulations
    
    trunc_array=np.ones(2*N)*np.sqrt(trunc) #excluded volume strength for each monomer
    trunc_array[N+R//2:2*N-R//2]=0 #unreplicated monomers don't have excluded volume interactions
    # create the reporter class
    reporter = HDF5Reporter(folder=folder, max_data_length=100) #this way one file corresponds to one timecourse

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

        all_bonds=[(i,i+N) for i in range(R//2,N-R//2)] #unreplicated monomers
        if forks_bound:
            all_bonds.append((R//2, N-R//2))

        # -----------Adding forces ---------------
        a.add_force(
            forcekits.polymer_chains(
                a,
                chains=[(0, N, True), (N, 2*N, True)], #Two chains for two full chromosomes
                bond_force_func=forces.harmonic_bonds, # adds harmonic bonds for polymers
                bond_force_kwargs={
                    'bondLength':length_array,
                    'bondWiggleDistance':wiggle_array, # Bond distance will fluctuate this much
                    # note factor of sqrt(2); so that before replication we don't have double the spring constant    
                },

                angle_force_func=forces.angle_force,
                angle_force_kwargs={
                    'k':0.05 # we are making a very flexible polymer, basically not necessary here
                },
                nonbonded_force_func=forces.polynomial_repulsive_with_exclusions, # this is the excluded volume potential
                nonbonded_force_kwargs={
                    #'trunc': trunc, # this will let chains cross sometimes (energy in kb*T units)
                    'trunc': trunc_array,
                    'radiusMult':1.05, # this is from old code
                    #'trunc':10.0, # this will resolve chain crossings and will not let chain cross anymore
                },
                except_bonds=True,
                extra_bonds=all_bonds
            )
        )

        a.add_force(forces.cylindrical_confinement(a,radius,bottom=-height/2,k=10*F_z,top=height/2))
        if num_tethers==2:
            a.add_force(forces.tether_particles(a,[0,N],k=[0,0,F_z],positions=[pos_old, pos_new])) #start with unreplicated; both oris below
        elif num_tethers==1:
            a.add_force(forces.tether_particles(a,[0],k=[0,0,F_z],positions=[pos_old])) #start with unreplicated; both oris below
        else:
            print("\nNot adding tethers")
        a.step = block

        # Minimize energy for first bonds
        print("Polymer burn-in")
        a.local_energy_minimization() 
        #Docs say first steps after energy minimization have large error; don't save
        a.integrator.step(40*steps_per_sample)

        a.step=block

        # Iterate over simulation time steps within each BondUpdater 
        for i in range(restartConfigurationEvery-2):
            # BondUpdater updates bonds at each step.
            if i % saveEveryConfigs == 0: #save just before resetting configuration  
                # save SMC, fork and monomer positions
                a.do_block(steps=steps_per_sample, save_extras={"SMC_step":a.step}) 
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

