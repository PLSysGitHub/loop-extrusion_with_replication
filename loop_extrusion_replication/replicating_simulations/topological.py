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

from ..translocators.topological  import topologicalSmcForkTranslocator
from ..confinement_volumes.c_crescentus import *

def calc_repl_pos(i,N):
    """
    For replicated segment of length n, calculate the position of replicated counterpart of position i
    """
    return i+N

def get_kb_from_theory(num_extruders):
    d = 358 # kb
    v3 = 0.46
    v2 = 0.59
    v1 = 0.83
    n_encounters = 850*(0.4*num_extruders/1371) 
    tb = d/n_encounters/v3 - d/n_encounters/v2*(1-n_encounters/d)
    kb = 1/tb
    return kb

class simulationBondUpdater(object):
    """
    This class precomputes simulation bonds for faster dynamic allocation. 
    """

    def __init__(self,N,N_1D, monomer_size,smcTransObject, trunc, num_tethers):
        """
        :param N: number of monomers in the polymer
        :param N_1D: number of monomers in the 1D polymer
        :param smcTransObject: smc translocator object to work with
        :param trunc: trunc energy for bonds
        :param monomer_size: size of monomer in nm
        :param num_tethers: number of tethers to hold ori(s) in place
        Arrays store data from all sampled time points
        """
        self.N=N
        self.num_tethers=num_tethers
        self.N_1D=N_1D
        self.smcObject = smcTransObject
        self.trunc=trunc
        self.allBonds = []
        self.smcs = []
        self.forkpos = []
        self.monomer_size=monomer_size

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
        self.replicatedParamDict = replicatedParamDict #chance polymer bond to this after replication; essentially double spring constant (because before replication two copies of spring in parallel)

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
        smcs =[]

        for dummy in range(blocks):
            if dummy>1: #first sample before steps, fork hasn't moved
                self.smcObject.steps(smcStepsPerBlock)

            left, right= self.smcObject.getPairwiseSMCs()
            fork=self.smcObject.getFork()

            left=np.floor(left*self.N/self.N_1D).astype(int)
            right=np.floor(right*self.N/self.N_1D).astype(int)
            fork=np.floor(fork*self.N/self.N_1D).astype(int)
            
            # add SMC bonds
            bonds = [(i, j) for i,j in zip(left, right)]
            replication_bonds=[(i,calc_repl_pos(i,self.N)) for i  in range(fork[0],fork[1]+1)] #tie unreplicated bit to old

            allBonds.append(bonds+replication_bonds)
            smcs.append(bonds)
            forks.append(fork)

        self.allBonds = allBonds
        self.smcs = smcs
        self.forkpos=forks
        self.uniqueBonds = list(set(sum(allBonds, [])))

        #adding forces and getting bond indices
        self.bondInds = []

        self.curBonds = allBonds.pop(0)
        self.curSmcs = smcs.pop(0)
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
        self.curSmcs=self.smcs.pop(0)
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

        #Check how far replication is, and update the confinement and ori positions
        N=self.N        
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
            self.excl.setParticleParameters(i+self.N,[np.sqrt(self.trunc)])

            if i>0: #avoid negative index; bond (N,0) dealt with in next for loop
                #debugging test of bond indices
                p1,p2,l,k=self.bondForce.getBondParameters(i)
                assert (p1==i-1 and p2==i), f"Error: got particles {p1} and {p2} for bond {i}!"
                self.bondForce.setBondParameters(i,i-1,i, **self.replicatedParamDict)
                p1,p2,l,k=self.bondForce.getBondParameters(i+self.N)
                assert (p1==i+self.N-1 and p2==i+self.N), f"Error: got particles {p1} and {p2} for bond {i+self.N}!"
                self.bondForce.setBondParameters(i+self.N,i+self.N-1,i+self.N, **self.replicatedParamDict)

        for i in range(self.curFork[1]+1,pastFork[1]+1):
            self.excl.setParticleParameters(i+self.N,[np.sqrt(self.trunc)])
            #debugging test of bond indices
            p1,p2,l,k=self.bondForce.getBondParameters(i+1)
            p1,p2=sorted([p1,p2])
            assert (p1==min(i,(i+1)%self.N) and p2==max(i,(i+1)%self.N)), f"Error: got particles {p1} and {p2} for bond {i+1}!"
            self.bondForce.setBondParameters(i+1,min(i,(i+1)%self.N), max(i,(i+1)%self.N), **self.replicatedParamDict) 

            p1,p2,l,k=self.bondForce.getBondParameters(i+self.N+1)
            p1,p2=sorted([p1,p2])
            assert (p1==min(i+self.N,(i+1)%self.N+self.N) and p2==max(i+self.N,(i+1)%self.N+self.N)), f"Error: got particles {p1} and {p2} for bond {i+1+self.N}!"
            self.bondForce.setBondParameters(i+1+self.N,min(i+self.N,(i+1)%self.N+self.N),max(i+self.N,(i+1)%self.N+self.N), **self.replicatedParamDict)

        self.bondForce.updateParametersInContext(context)  # now run this to update things in the context
        if self.num_tethers>0:
            self.tether.updateParametersInContext(context)
        self.excl.updateParametersInContext(context)

        return self.curSmcs, self.curFork #these can be saved for the configuration

def initModel(knockOffProb,kOriToTer_kTerToOri,parSsites,\
        wind=0.33,terSites=[2000,2050], terSiteStrength=0.0025,parSstrengths=None,\
        SEPARATION=200,LIFETIME = 2020/0.05,N = 4000,BASE_STOCHASTICITY=0.05, fork_rate=0.,stallFork=0):

    """    
    N : int
    number of sites on original chromosome

    knockOffProb : float
    The probability of dissociating from the chromosome if SMC is stalled

    kOriToTer_kTerToOri : 2-list of floats
    The probabilities of being knocked off the chromosome if stepping onto an occupied site in 
    the 1) Ori to ter or 2) Ter to ori directions. Note no bypassing for topological loop-extrusion.
    Typically, we let 1) and 2) be equal.

    parSsites : list of ints
    Degree locations for the parS loading sites

    parSstrengths : float
    Weight of loading at each parS sites versus other sites, where non-parS sites have 
    a relative strength of 1. The default value if None is specified is 16000/len(parSsites).

    wind : float
    Factor (between 0 and 1) by which to slow down when moving towards the ori

    terSites : list of ints
    List of monomers whereby to stall SMCs

    terSiteStrength : float
    List of monomers whereby to stall SMCs

    SEPARATION : int
    Average of monomer spacings between SMC complexes (i.e. related to SMC density)

    LIFETIME : float
    The number of simulation steps before the SMC complex dissociates spontaneously.
    Typically this value is 2000/BASE_STOCHASTICITY

    BASE_STOCHASTICITY: float
    The probability (in simulation steps) that an SMC subunit will move forward.

    fork_rate : float
    Rate of fork movement per simulation step

    stallFork : int
    Probability that SMC stalls when encounters a fork

    """  

    birthArray = np.ones(2*N, dtype=np.double)  
    deathArray = np.zeros(2*N, dtype=np.double) + 1. / LIFETIME 
    stallDeathArray = np.zeros(2*N, dtype=np.double) + knockOffProb
    stallLeftArray = np.zeros(2*N, dtype = np.double)
    stallRightArray = np.zeros(2*N, dtype = np.double)
    stepArrayR = BASE_STOCHASTICITY*np.ones(2*N, dtype=np.double) 
    stepArrayL = BASE_STOCHASTICITY*np.ones(2*N, dtype=np.double)

    deathList = [x for x in np.arange(terSites[0],terSites[1])]

    for i in deathList:
        deathArray[i] = 0.05*BASE_STOCHASTICITY
        deathArray[calc_repl_pos(i,N)]=0.05*BASE_STOCHASTICITY

    # set parS locations, note replicated segment has zero loading at this point
    if parSstrengths is None:
        parSstrengths = [16000/len(parSsites)]*len(parSsites)

    for s,ss in zip(parSsites,parSstrengths):
        s_ind=np.mod(int(s/360*N),N)  
        birthArray[s_ind] = ss

    #Before replication, zero loading on new chromosome
    birthArray[N:2*N]=0.

    # set baseline translocation asymmetry
    stepArrayL[0:terSites[0]] *= (1-wind) #decrease to ori
    stepArrayL[N:N+terSites[0]] *= (1-wind) #decrease to ori

    stepArrayR[terSites[1]:N] *= (1-wind) #increase to ori
    stepArrayR[N+terSites[1]:2*N] *= (1-wind) #increase to ori

    smcNum = int((N) / SEPARATION) #smc number per chromosome. density kept constant during sims as chromosome replicated
    pauseArrayR = 1-stepArrayR
    pauseArrayL = 1-stepArrayL
    SMCTran = topologicalSmcForkTranslocator(0, birthArray, deathArray, stallLeftArray, \
            stallRightArray, pauseArrayL, pauseArrayR, \
            stallDeathArray, kOriToTer_kTerToOri,  smcNum, fork_rate,stallFork)
    return SMCTran

def run_simulation(monomer_size, monomer_wig,knockOffProb, kOriToTer_kTerToOri, parSsites, wind, terSites,\
        terSiteStrength, parSstrengths, separation,lifetime, N,N_1D, BASE_STOCHASTICITY,\
        translocator_initialization_steps, steps_per_sample, radius, height, z_ori, \
        smcStepsPerBlock, smcBondDist, smcBondWiggleDist, save_folder, saveEveryConfigs, \
        savedSamples, restartConfigurationEvery, GPU_choice = 0, F_z=0., col_rate=0.1, trunc=0.5,\
        fork_rate=0.05, stallFork=0., top_monomer=0, num_tethers=0):
    """
    This function initiates the bacterial chromosome. 

    monomer_size : float
    The size of the monomers in nm

    monomer_wig : float
    The size of the monomer wiggle in sim units
    
    knockOffProb : float
    The probability of dissociating from the chromosome if SMC is stalled

    kOriToTer_kTerToOri: 2-list of floats
    The probabilities of being knocked off the chromosome if stepping onto an occupied site in 
    the 1) Ori to ter or 2) Ter to ori directions. Typically, we let 1) and 2) be equal.

    parSsites : list of ints
    Degree locations for the parS loading sites

    parSstrengths : float
    Weight of loading at each parS sites versus other sites, where non-parS sites have 
    a relative strength of 1. The default value if None is specified is 16000/len(parSsites).

    wind : float
    Factor (between 0 and 1) by which to slow down when moving towards the ori

    terSites : list of ints
    List of monomers whereby to stall SMCs

    terSiteStrength : float
    List of monomers whereby to stall SMCs

    SEPARATION : int
    Average of monomer spacings between SMC complexes (i.e. related to SMC density)

    LIFETIME : float
    The number of simulation steps before the SMC complex dissociates spontaneously.
    Typically this value is 2000/BASE_STOCHASTICITY

    N : int
    Number of monomers in the chromosome

    N_1D : int
    Number of monomers in the chromosome in 1D

    BASE_STOCHASTICITY: float
    The probability (in simulation steps) that an SMC subunit will move forward.

    steps_per_sample : int
    The number of  polymer simulation integration steps to be carried out per block.

    radius : float
    The radius of the confinement in sim units

    height : float
    The height of the confinement in sim units

    z_ori : float
    The z-position the ori is tethered to in sim units

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
    Energy scale for excluded volume interactions

    fork_rate : float
    The probability of a fork to move forward

    stallFork : float
    The probability that a LE stalls at the fork

    top_monomer : int
    The monomer number at which to tether the top of the polymer, and which to initialise at the pole

    num_tethers : int
    The number of tethered ori(s), between 0 and 2
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
        SMCTran = initModel(knockOffProb, kOriToTer_kTerToOri, parSsites,\
        wind, terSites, terSiteStrength, parSstrengths,\
        separation,lifetime, N_1D, BASE_STOCHASTICITY, fork_rate,stallFork)

        #Distribute SMCs on first chromosome, then start replication fork
        SMCTran.steps(translocator_initialization_steps) #this is without stalling at fork
        SMCTran.start_replication() #start stalling at the fork, start moving fork

        #Now feed bond generators to BondUpdater 
        BondUpdater = simulationBondUpdater(N,N_1D,monomer_size,SMCTran, trunc, num_tethers)

        # doSave = BondUpdaterCount >= InitsSkip
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
                    'trunc': np.concatenate((np.sqrt(trunc)*np.ones(N+1),np.zeros(N-1))),
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
        print("Topological loop extrusion simulations")
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
        smcs, fork = BondUpdater.step(a.context) #get first bonds, update context
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
            smcs, fork = BondUpdater.step(a.context)
            if i % saveEveryConfigs == 0: #save just before resetting configuration  
                # save SMC, fork and monomer positions
                a.do_block(steps=steps_per_sample, save_extras={"SMCs":smcs,"Fork":fork, "SMC_step":a.step}) 
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

