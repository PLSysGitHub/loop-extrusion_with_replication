import os
import time
import numpy as np
import polychrom
from polychrom import polymerutils
from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file
import openmm 
import shutil
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},reload_support=True)
import warnings
import h5py
import glob
import re

from ..bacterial_species import cell

def run_simulation(bacterium, monomer_wig,\
        steps_per_sample, save_folder, smcSteps, numSims,\
        saveEveryConfigs, GPU_choice = 0, F_z=0., add_tether=False,mass=100,\
        col_rate=0.1, trunc=0.5, top_monomer=0, no_confinement=False,infinite_tube=False):
    """
    Run simulations of chromosome in flat confinement with loop extrusion.

    Args:
        smcTrajFolder (string) : folder where 1D sims of LE were saved
        bacterium : container for details on bacterium's size and loop-extrusion properties
        monomer_wig (float): wiggle distance for monomers
        steps_per_sample (int): polymer simulation steps for SMC update
        smcBondDist (float): distance between SMCs
        smcBondWiggleDist (float): wiggle distance for SMCs
        save_folder (str): folder to save the 3D simulation data
        savedSamples (int): number of saved samples
        GPU_choice (int): GPU number
        F_z (float): force to pull the ends of the polymer and to confine it
        col_rate (float): collision rate. sets drag
        trunc (float) : truncation radius for the excluded volume potential
        top_monomer (int) : which monomer is at the pole for initial config
    Returns:
        None
    """
    N=bacterium.N
    polymerSteps= smcSteps*steps_per_sample

    savedSamples= smcSteps//saveEveryConfigs
    print(f"Simulating a polymer of {N} monomers")
    block=0 #start count from zero   

    # clean up the simulation directory
    folder = save_folder
    if os.path.exists(folder):
        shutil.rmtree(folder)

    # create the reporter class
    reporter = HDF5Reporter(folder=folder, max_data_length=150) 

    # Iterate over various BondUpdaterInitializations
    for BondUpdaterCount in range(numSims):
        # create the starting conformation
        start_data =bacterium.start_point_unreplicating(top_monomer) #unreplicated chromosome spread across cell

        # simulation parameters are defined below 
        a = Simulation(
                platform="cuda",
                integrator="variableLangevin", 
                error_tol=0.001,
                GPU = "{}".format(GPU_choice), 
                collision_rate=col_rate, 
                N = len(start_data),
                max_Ek=100.,
                mass=mass,
                reporters=[reporter],
                PBCbox=False,
                precision="mixed",
                verbose=False)  # timestep not necessary for variableLangevin

        a.set_data(start_data)  # loads polymer.

        # -----------Adding forces ---------------
        a.add_force(
            forcekits.polymer_chains(
                a,
                chains=[(0, N, True)], #circular chromosome
                bond_force_func=forces.harmonic_bonds, # adds harmonic bonds for polymers
                bond_force_kwargs={
                    'bondLength':1.0, # Bond length
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

        #Initialize confinement.
        if infinite_tube:
            a.add_force(forces.rounded_cap_cylindrical_confinement(a,bacterium.radius,bottom=None,k=10*F_z))
        elif not no_confinement:
            a.add_force(forces.rounded_cap_cylindrical_confinement(a,bacterium.radius,bottom=-bacterium.L_0/2,k=10*F_z,top=bacterium.L_0/2))

        if add_tether:
            a.add_force(forces.tether_particles(a,[top_monomer],k=[0,0,F_z],positions=[[0,0,-bacterium.L_0/2+2]]))#tether one monomer in place 

        a.step = block

        # Minimize energy for first bonds
        print("Polymer burn-in")
        a.local_energy_minimization() 
        
        #Docs say first steps after energy minimization have large error; don't save
        a.integrator.step(500*steps_per_sample)
        a.step=block

        # Iterate over simulation time steps within each BondUpdater 
        for i in range(smcSteps-2):
            if i % saveEveryConfigs == 0: #3D polymer steps plus save 
                a.do_block(steps=steps_per_sample, save_extras={"simulation_run":BondUpdaterCount}) 
            else:
                a.integrator.step(steps_per_sample)  #do 3D steps without getting the positions from the GPU (faster)

        block = a.step
        del a

        time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)

    # dump data to output file
    reporter.dump_data()
    done_file = open(os.path.join(folder,'sim_done.txt'),"w+")
    done_file.close()
    del reporter
