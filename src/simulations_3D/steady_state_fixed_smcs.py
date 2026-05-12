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

def num_time_steps_trajectories(folder_name):
    files=os.listdir(folder_name)
    filtered_files = [os.path.join(folder_name,f) for f in files if "simulation_" in f]
    
    with h5py.File(filtered_files[0], 'r') as f:
        ts=f['ts'][:]

    nt=ts.size
    nsims=len(filtered_files)

    return nt, nsims, filtered_files

def load_smcs_from_h5(filename, ind):
    #array of vectors. Each vector is a time-step, each item is a leg position
    with h5py.File(filename, 'r') as f:
        l_sites = f['l_sites'][ind]
        r_sites = f['r_sites'][ind]

    #for 3D polymer simulations, we need to have two vectors; one with left legs, one with right legs.
    # but the simulations might have some indices that are -1, corresponding to unbound SMCs. filter out
    are_bound=l_sites>=0

    return l_sites[are_bound], r_sites[are_bound]

class simulationBondUpdater(object):
    """
    This class precomputes simulation bonds for faster dynamic allocation. 
    """

    def __init__(self,N,smcTrajectoryFile,which_step):
        """
        :param N: number of monomers in the polymer
        :param smcTransObject: smc translocator object to work with
        :param trunc: excluded volume strength
        Arrays store data from all sampled time points
        """
        self.N=N
        self.smcFile = smcTrajectoryFile
        self.allBonds = []
        self.smcs = []
        self.smc_step=which_step

    def setParams(self, activeParamDict, inactiveParamDict):
        """
        A method to set parameters for bonds.
        It is a separate method because you may want to have a Simulation object already existing

        :param activeParamDict: a dict (argument:value) of addBond arguments for active bonds
        :param inactiveParamDict:  a dict (argument:value) of addBond arguments for inactive bonds
        """
        self.activeParamDict = activeParamDict #active condensin or pre-replication bond
        self.inactiveParamDict = inactiveParamDict #no condensin or pre-replication bond

    def LEF_simulation(self, bondForce, smcTimeSteps = 2):
        """
        A method that milks smcTranslocator object
        and creates a set of unique bonds, etc.

        :param bondForce: a bondforce object (new after simulation restart!)
        :param blocks: number of blocks to precalculate
        :param smcStepsPerBlock: number of smcTranslocator steps per block
        :return:
        """
        if len(self.allBonds) != 0:
            raise ValueError("Not all bonds were used; {0} sets left".format(len(self.allBonds)))


        self.bondForce = bondForce

        allBonds = []
        smcs =[]

        for timeStep in range(smcTimeSteps):
            left, right=load_smcs_from_h5(self.smcFile, self.smc_step) 
            
            # add SMC bonds
            bonds = [(int(i), int(j)) for i,j in zip(left, right)]

            allBonds.append(bonds)
            smcs.append(bonds)

        self.allBonds = allBonds
        self.smcs = smcs
        self.uniqueBonds = list(set(sum(allBonds, [])))

        #adding forces and getting bond indices
        self.bondInds = []

        self.curBonds = allBonds.pop(0)

        for bond in self.uniqueBonds:
            paramset = self.activeParamDict if (bond in self.curBonds) else self.inactiveParamDict
            ind = self.bondForce.addBond(bond[0], bond[1], **paramset)
            self.bondInds.append(ind)

        self.bondToInd = {i:j for i,j in zip(self.uniqueBonds, self.bondInds)}

        return self.curBonds


    def step(self, context, verbose=False):
        """
        Update the bonds to the next step.
        It sets bonds for you automatically!
        :param context:  context
        :return: (current bonds, previous step bonds); just for reference
        """
        if len(self.allBonds) == 0:
            raise ValueError("No bonds left to run; you should restart simulation")

        pastBonds = self.curBonds
        #Get parameters for next time step
        self.curBonds = self.allBonds.pop(0)
        self.curSmcs=self.smcs.pop(0)

        #Change bonds
        bondsRemove = [i for i in pastBonds if i not in self.curBonds]
        bondsAdd = [i for i in self.curBonds if i not in pastBonds]
        bondsStay = [i for i in pastBonds if i in self.curBonds]
        if verbose:
            print("{0} bonds stay, {1} new bonds, {2} bonds removed".format(len(bondsStay),len(bondsAdd), len(bondsRemove)))
        bondsToChange = bondsAdd + bondsRemove
        bondsIsAdd = [True] * len(bondsAdd) + [False] * len(bondsRemove)

        for bond, isAdd in zip(bondsToChange, bondsIsAdd):
            if bond in self.bondToInd.keys():
                ind = self.bondToInd[bond] 
                paramset = self.activeParamDict if isAdd else self.inactiveParamDict
                self.bondForce.setBondParameters(ind, bond[0], bond[1], **paramset)  # actually updating bonds
            else:
                print("Key ", bond, "not found!")    


        self.bondForce.updateParametersInContext(context)  # now run this to update things in the context

        return self.curSmcs #these can be saved for the configuration

def run_simulation(smcTrajFolder, sim_step_ind, bacterium, monomer_wig,\
        steps_per_sample, smcBondDist, smcBondWiggleDist, save_folder,\
        saveEveryConfigs, GPU_choice = 0, F_z=0., add_tether=False,\
        col_rate=0.1, trunc=0.5, top_monomer=0, no_confinement=False,infinite_tube=False, platform="cuda"):
    """
    Run simulations of chromosome in flat confinement with loop extrusion.

    Args:
        smcTrajFolder (string) : folder where 1D sims of LE were saved
        sim_step_ind (int) : stores which LE config we should take each round
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
    smcSteps, numSims, trajectoryFiles=num_time_steps_trajectories(smcTrajFolder)
    N=bacterium.N
    polymerSteps= smcSteps*steps_per_sample

    savedSamples= smcSteps//saveEveryConfigs
    print(f"Got {smcSteps} steps and {numSims} sims.\nSaving {savedSamples} 3D configurations per trajectory.")
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
        smcFile=trajectoryFiles[BondUpdaterCount]

        # create the starting conformation
        start_data =bacterium.start_point_unreplicating(top_monomer) #unreplicated chromosome spread across cell

        #Now feed bond generators to BondUpdater 
        BondUpdater = simulationBondUpdater(N, smcFile, sim_step_ind) #loads the LEs at a given step

        # simulation parameters are defined below 
        a = Simulation(
                platform=platform,
                integrator="variableLangevin", 
                error_tol=0.001,
                GPU = "{}".format(GPU_choice), 
                collision_rate=col_rate, 
                N = len(start_data),
                max_Ek=100.,
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

        # -----------Initialize bond updater. Add bonds ---------------
        a.step = block
        kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
        kbond_replicated = a.kbondScalingFactor /(monomer_wig**2) #update to this after replicating monomer
        bondDist = smcBondDist * a.length_scale
        activeParams = {"length":bondDist,"k":kbond}
        inactiveParams = {"length":bondDist, "k":0}
        replicatedParams = {"length":1, "k":kbond_replicated}

        #Pass the forces and parameters to the BondUpdater
        BondUpdater.setParams(activeParams, inactiveParams)

        #fetch only the first loop extruder positions
        print("Load loop-extrusion data")
        BondUpdater.LEF_simulation(a.force_dict['harmonic_bonds'],smcTimeSteps=2) 

        # Minimize energy for first bonds
        print("Polymer burn-in")
        a.local_energy_minimization() 
        
        smcs = BondUpdater.step(a.context) #get first bonds, update context
        #Docs say first steps after energy minimization have large error; don't save
        a.integrator.step(80*steps_per_sample)
        
        a.step=block

        # Iterate over simulation time steps within each BondUpdater. Don't update SMC positions! 
        for i in range(smcSteps-2):
            if i % saveEveryConfigs == 0: #3D polymer steps plus save 
                a.do_block(steps=steps_per_sample, save_extras={"SMCs":smcs, "simulation_run":BondUpdaterCount}) 
            else:
                a.integrator.step(steps_per_sample)  #do 3D steps without getting the positions from the GPU (faster)

        block = a.step
        del a
        del BondUpdater

        time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)

    # dump data to output file
    reporter.dump_data()
    done_file = open(os.path.join(folder,'sim_done.txt'),"w+")
    done_file.close()
    del reporter
