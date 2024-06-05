import pickle
import os
import time
import numpy as np
import polychrom
from polychrom import polymerutils
from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.starting_conformations import grow_cubic
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

from ..translocators.no_smcs  import forkTranslocator
from ..confinement_volumes.c_crescentus import *

def calc_repl_pos(i,N):
    """
    For replicated segment of length n, calculate the position of replicated counterpart of position i
    """
    return i+N


class simulationBondUpdater(object):
    """
    This class precomputes simulation bonds for faster dynamic allocation. 
    """

    def __init__(self,N,N_1D,monomer_size, smcTransObject,num_tethers, tie_forks):
        """
        :param smcTransObject: smc translocator object to work with
        :param N: number of monomers in the polymer
        :param N_1D: number of monomers in the 1D simulations
        :param num_tethers: number of tethers to hold ori(s) in place
        Arrays store data from all sampled time points
        """
        self.smcObject = smcTransObject
        self.N=N
        self.N_1D=N_1D
        self.allBonds = []
        self.forkpos = []
        self.monomer_size=monomer_size
        self.num_tethers=num_tethers
        self.tie_forks=tie_forks

    def setParams(self, activeParamDict, inactiveParamDict, replicatedParamDict):
        """
        A method to set parameters for bonds.
        It is a separate method because you may want to have a Simulation object already existing

        :param activeParamDict: a dict (argument:value) of addBond arguments for active bonds
        :param inactiveParamDict:  a dict (argument:value) of addBond arguments for inactive bonds
        :param replicatedParamDict:  a dict (argument:value) of addBond arguments for replicated bonds
        """
        self.activeParamDict = activeParamDict #active condensin or pre-replication bond
        self.inactiveParamDict = inactiveParamDict #no condensin or pre-replication bond
        self.replicatedParamDict = replicatedParamDict #change polymer bond to this after replication; essentially double spring constant (because before replication two copies of spring in parallel)

    def LEF_simulation(self, bondForce, cylinder, tether, excl, smcStepsPerBlock,  blocks = 100):
        """
        A method that milks smcTranslocator object
        and creates a set of unique bonds, etc.

        :param bondForce: a bondforce object (new after simulation restart!)
        :param cylinder: the cylindrical confinement force object
        :param tether: tether force object that holds oris in place
        :param excl: excluded volume force object
        :param blocks: number of blocks to precalculate
        :param smcStepsPerBlock: number of smcTranslocator steps per block
        :return:
        """
        if len(self.allBonds) != 0:
            raise ValueError("Not all bonds were used; {0} sets left".format(len(self.allBonds)))


        self.bondForce = bondForce
        self.cylinder = cylinder
        self.tether = tether
        self.excl = excl

        allBonds = []
        forks=[]

        for dummy in range(blocks):
            if dummy>1: #first sample before steps, fork hasn't moved
                self.smcObject.steps(smcStepsPerBlock)

            fork=self.smcObject.getFork()
            fork=np.floor(fork*self.N/self.N_1D).astype(int)
            
            replication_bonds=[(i,calc_repl_pos(i,self.N)) for i  in range(fork[0],fork[1]+1)] #tie unreplicated bit to old

            if self.tie_forks:
                replication_bonds.append((fork[0],fork[1]))
            allBonds.append(replication_bonds)
            forks.append(fork)

        self.allBonds = allBonds
        self.forkpos=forks
        self.uniqueBonds = list(set(sum(allBonds, [])))

        #adding forces and getting bond indices
        self.bondInds = []

        self.curBonds = allBonds.pop(0)
        self.curFork = forks.pop(0)

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
        pastFork = self.curFork
        #Get parameters for next time step
        self.curBonds = self.allBonds.pop(0)
        self.curFork=self.forkpos.pop(0)

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

        N=self.N
        #Check how far replication is, and update the confinement and ori positions
        n=self.curFork[0]+N-self.curFork[1]
        
        height=R_to_height(n,N, self.monomer_size)
        if self.num_tethers==2:
            old_ori, new_ori=R_to_z_oris(n ,N, height, self.monomer_size)
            self.tether.setParticleParameters(0,0,old_ori)
            self.tether.setParticleParameters(1,1,new_ori)
        elif self.num_tethers==1:
            old_ori, new_ori=R_to_z_oris(n ,N, height, self.monomer_size)
            self.tether.setParticleParameters(0,0,old_ori)

        context.setParameter("cylindrical_confinement_top",height/2) #grow both sides by updating both
        context.setParameter("cylindrical_confinement_bottom",-height/2)

        # Turn on excluded volume and increase spring constant for replicated monomers
        for i in range(pastFork[0],self.curFork[0]):
            self.excl.setParticleParameters(i+N,self.excl.getParticleParameters(i))

            if i>0: #avoid negative index; bond (N,0) dealt with in next for loop
                #debugging test of bond indices
                p1,p2,l,k=self.bondForce.getBondParameters(i)
                assert (p1==i-1 and p2==i), f"Error: got particles {p1} and {p2} for bond {i}!"
                self.bondForce.setBondParameters(i,i-1,i, **self.replicatedParamDict)
                p1,p2,l,k=self.bondForce.getBondParameters(i+N)
                assert (p1==i+N-1 and p2==i+N), f"Error: got particles {p1} and {p2} for bond {i+N}!"
                self.bondForce.setBondParameters(i+N,i+N-1,i+N, **self.replicatedParamDict)

        for i in range(self.curFork[1]+1,pastFork[1]+1):
            self.excl.setParticleParameters(i+N,self.excl.getParticleParameters(i))
            #debugging test of bond indices
            p1,p2,l,k=self.bondForce.getBondParameters(i+1)
            p1,p2=sorted([p1,p2])
            assert (p1==min(i,(i+1)%N) and p2==max(i,(i+1)%N)), f"Error: got particles {p1} and {p2} for bond {i+1}!"
            self.bondForce.setBondParameters(i+1,min(i,(i+1)%N), max(i,(i+1)%N), **self.replicatedParamDict) 

            p1,p2,l,k=self.bondForce.getBondParameters(i+N+1)
            p1,p2=sorted([p1,p2])
            assert (p1==min(i+N,(i+1)%N+N) and p2==max(i+N,(i+1)%N+N)), f"Error: got particles {p1} and {p2} for bond {i+1+N}!"
            self.bondForce.setBondParameters(i+1+N,min(i+N,(i+1)%N+N),max(i+N,(i+1)%N+N), **self.replicatedParamDict)

        self.bondForce.updateParametersInContext(context)  # now run this to update things in the context
        if self.num_tethers>0:
            self.tether.updateParametersInContext(context)
        self.excl.updateParametersInContext(context)

        return self.curFork #these can be saved for the configuration

def initModel(N = 4000, fork_rate=0.):

    """    
    N : int
    number of 1D sites on original chromosome

    fork_rate : float
    rate of fork movement in monomer units per time step
    """  

    SMCTran = forkTranslocator(N,0,fork_rate)
    return SMCTran

def run_simulation(monomer_size, monomer_wig,N,N_1D,steps_per_sample, radius, height, z_ori, \
        smcStepsPerBlock, smcBondDist, smcBondWiggleDist, save_folder, saveEveryConfigs, \
        savedSamples, restartConfigurationEvery, GPU_choice = 0, F_z=0., col_rate=0.1, \
        trunc=0.5,fork_rate=0.05, top_monomer=0, num_tethers=2, tie_forks=False):
    """
    This function initiates the bacterial chromosome. 

    monomer_size : float
    The size of the monomer in nanometers

    monomer_wig : float
    The wiggle distance of the monomer in sim units

    N : int
    The number of monomers in the 3D polymer

    N_1D : int
    The number of monomers in the 1D polymer

    steps_per_sample : int
    The number of  polymer simulation integration steps to be carried out per block.

    radius : float
    The radius of the cylindrical confinement in nanometers

    height : float
    The height of the cylindrical confinement in nanometers

    z_ori : float
    The height to which oris are tethered (if tethered)

    smcStepsPerBlock : int
    Number of SMC translocator steps taken per polymer simulation time step

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
    The energy scale for the Lennard-Jones potential

    fork_rate : float
    The rate of fork movement in monomer units per time step

    top_monomer : int
    The monomer at the top of the polymer (ie the monomer that is tethered)

    num_tethers : int
    The number of oris that is tethered; between 0 and 2
    """

    # assertions for easy managing code below 
    assert restartConfigurationEvery % saveEveryConfigs == 0 
    assert (savedSamples * saveEveryConfigs) % restartConfigurationEvery == 0 

    savesPerInit = restartConfigurationEvery // saveEveryConfigs
    # InitsSkip = saveEveryConfigs * skipSamples  // restartConfigurationEvery
    InitsSave = savedSamples * saveEveryConfigs // restartConfigurationEvery
    InitsTotal  = InitsSave #+ InitsSkip
    print("BondUpdater will be initialized {0} times.".format(InitsTotal))    

    pulled=top_monomer+N
    block=0 #start count from zero   

    # clean up the simulation directory
    folder = save_folder
    if os.path.exists(folder):
        shutil.rmtree(folder)

    start_data = start_point_unsegregated(top_monomer,0,N,height)

    # create the reporter class
    reporter = HDF5Reporter(folder=folder, max_data_length=savesPerInit) #this way one file corresponds to one timecourse

    # Iterate over various BondUpdaterInitializations
    for BondUpdaterCount in range(InitsTotal):

        # create SMC translocation object
        SMCTran = initModel(N_1D, fork_rate)

        #Distribute SMCs on first chromosome, then start replication fork
        SMCTran.start_replication() #start stalling at the fork, start moving fork

        #Now feed bond generators to BondUpdater 
        BondUpdater = simulationBondUpdater(N,N_1D,monomer_size,SMCTran, num_tethers, tie_forks)

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
                chains=[(0, N, True), (N, 2*N, True)], #Two chains for two full chromosomes
                bond_force_func=forces.harmonic_bonds, # adds harmonic bonds for polymers
                bond_force_kwargs={
                    'bondLength':1,
                    'bondWiggleDistance':monomer_wig*np.sqrt(2), # Bond distance will fluctuate this much
                    # note factor of sqrt(2); so that before replication we don't have double the spring constant    
                },

                angle_force_func=forces.angle_force,
                angle_force_kwargs={
                    'k':0.05 # we are making a very flexible polymer, basically not necessary here
                },
                nonbonded_force_func=forces.polynomial_repulsive_with_exclusions, # this is the excluded volume potential
                nonbonded_force_kwargs={
                    #'trunc': trunc, # this will let chains cross sometimes (energy in kb*T units)
                    'trunc': np.concatenate((np.sqrt(trunc)*np.ones(N+1),np.zeros(N-1))), #unreplicated monomers dont have excluded volume
                    'radiusMult':1.05, # this is from old code
                    #'trunc':10.0, # this will resolve chain crossings and will not let chain cross anymore
                },
                except_bonds=True,
                extra_bonds=[(N//2,calc_repl_pos(N//2,N))] #termini always connected
            )
        )

        a.add_force(forces.cylindrical_confinement(a,radius,bottom=-height/2,k=10*F_z,top=height/2))
        if num_tethers==2:
            a.add_force(forces.tether_particles(a,[top_monomer,pulled],k=[0,0,F_z],positions=[[0,0,-z_ori],[0,0,-z_ori]])) #start with unreplicated; both oris below
        elif num_tethers==1:
            a.add_force(forces.tether_particles(a,[top_monomer],k=[0,0,F_z],positions=[[0,0,-z_ori]])) #only one ori tethered
        
        # -----------Initialize bond updater. Add bonds ---------------
        a.step = block
        kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
        kbond_replicated = a.kbondScalingFactor /(monomer_wig**2) #update to this after replicating monomer
        bondDist = smcBondDist * a.length_scale
        activeParams = {"length":bondDist,"k":kbond}
        inactiveParams = {"length":bondDist, "k":0}
        replicatedParams = {"length":1, "k":kbond_replicated}

        #Pass the forces and parameters to the BondUpdater
        BondUpdater.setParams(activeParams, inactiveParams, replicatedParams)

        #Perform LEF simulation, sample bonds and fetch the first ones
        print("Fork position simulations")
        if num_tethers==0:
            BondUpdater.LEF_simulation(a.force_dict['harmonic_bonds'],a.force_dict['cylindrical_confinement'],
                None, a.force_dict['polynomial_repulsive_with_exclusions'],
                smcStepsPerBlock=smcStepsPerBlock,blocks=restartConfigurationEvery) 
        else:
            BondUpdater.LEF_simulation(a.force_dict['harmonic_bonds'],a.force_dict['cylindrical_confinement'],
                a.force_dict['Tethers'], a.force_dict['polynomial_repulsive_with_exclusions'],
                smcStepsPerBlock=smcStepsPerBlock,blocks=restartConfigurationEvery) 

        # Minimize energy for first bonds
        print("Polymer burn-in")
        a.local_energy_minimization() 
        fork = BondUpdater.step(a.context) #get first bonds, update context
        #Docs say first steps after energy minimization have large error; don't save
        a.integrator.step(40*steps_per_sample)

        print("Storing initial configuration")
        start_data=a.get_data() #start_configuration for next round is unreplicated and roughly in cylinder of right size
        a.step=block

        # Check ori positions for debugging:
        pos1=a.get_data()[0,2]
        pos2=a.get_data()[N,2]
        print("After energy min orig at ",pos1," new at ",pos2)

        # Iterate over simulation time steps within each BondUpdater 
        for i in range(restartConfigurationEvery-2):
            # BondUpdater updates bonds at each step.
            fork = BondUpdater.step(a.context)
            if i % saveEveryConfigs == 0: #save just before resetting configuration  
                # save SMC, fork and monomer positions
                a.do_block(steps=steps_per_sample, save_extras={"Fork":fork, "SMC_step":a.step}) 
            else:
                a.integrator.step(steps_per_sample)  #do steps without getting the positions from the GPU (faster)

        block = a.step
        del a
        del BondUpdater

        time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)

    # dump data to output file
    reporter.dump_data()
    done_file = open(os.path.join(folder,'sim_done.txt'),"w+")
    done_file.close()
    del reporter

