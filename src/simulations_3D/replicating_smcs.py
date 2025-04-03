import os
from tqdm import tqdm
import time
import numpy as np
import polychrom
import simtk.unit
from polychrom import polymerutils
from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file
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
    """
    Get the number of time steps and number of simulations from a folder with SMC trajectories
    :param folder_name: folder with SMC trajectories
    :return: number of time steps, number of simulations, list of files
    """
    files=os.listdir(folder_name)
    filtered_files = [os.path.join(folder_name,f) for f in files if "simulation_" in f]
    
    with h5py.File(filtered_files[0], 'r') as f:
        ts=f['ts'][:]

    nt=ts.size
    nsims=len(filtered_files)

    return nt, nsims, filtered_files

def load_smcs_forks_from_h5(filename, t):
    """
    Load SMCs and forks from a h5 file
    :param filename: h5 file with SMCs and forks
    :param ind: index of the time step
    :return: left, right, fork positions
    """

    #array of vectors. Each vector is a time-step, each item is a leg position
    with h5py.File(filename, 'r') as f:
        times=f["ts"][:]
        if times[0]<t:
            ind=np.where(times<t)[0][-1]
        else:
            ind=0        
        l_sites = f['l_sites'][ind]
        r_sites = f['r_sites'][ind]
        forks = f['forks'][ind]

    #for 3D polymer simulations, we need to have two vectors; one with left legs, one with right legs.
    # but the simulations might have some indices that are -1, corresponding to unbound SMCs. filter out
    are_bound=l_sites>=0

    return l_sites[are_bound], r_sites[are_bound], forks

class simulationBondUpdater(object):
    """
    This class precomputes simulation bonds for faster dynamic allocation. 
    """

    def __init__(self,bacterium, smcTrajectoryFile, trunc, num_tethers, dt=1/60):
        """
        Initialize the bond updater object
        :param bacterium: a cell type object
        :param smcTrajectoryFile: a file with SMC trajectories
        :param trunc: excluded volume strength
        :param num_tethers: number of tethers
        :param dt: time step in minutes, for calculating confinement dimensions
        Arrays store data from all sampled time points
        """
        self.bacterium=bacterium
        self.N=bacterium.N
        self.smcFile = smcTrajectoryFile
        self.trunc=trunc
        self.allBonds = []
        self.smcs = []
        self.forkpos= []
        self.num_tethers=num_tethers
        self.dt=dt
        self.current_time=0

    def setParams(self, activeParamDict, inactiveParamDict):
        """
        A method to set parameters for bonds.
        It is a separate method because you may want to have a Simulation object already existing

        :param activeParamDict: a dict (argument:value) of addBond arguments for active bonds
        :param inactiveParamDict:  a dict (argument:value) of addBond arguments for inactive bonds
        """
        self.activeParamDict = activeParamDict #active condensin or pre-replication bond
        self.inactiveParamDict = inactiveParamDict #no condensin or pre-replication bond

    def LEF_simulation(self, bondForce, cylinder, excl, tether, smcTimeSteps = 100):
        """
        A method that milks smcTranslocator object
        and creates a set of unique bonds, etc.

        :param bondForce: a bondforce object (new after simulation restart!)
        :param cylinder: a cylindrical confinement object
        :param excl: an excluded volume object
        :param tether: a tether object
        :param smcTimeSteps: number of time steps in the SMC trajectory
        :return:
        """
        if len(self.allBonds) != 0:
            raise ValueError("Not all bonds were used; {0} sets left".format(len(self.allBonds)))


        self.bondForce = bondForce
        self.cylinder = cylinder
        self.excl = excl
        self.tether= tether

        allBonds = []
        smcs =[]
        forks = []

        for timeStep in range(smcTimeSteps):
            t=timeStep*self.dt
            left, right, fork =load_smcs_forks_from_h5(self.smcFile, t) 
            fork=np.sort(fork)
            
            # add SMC bonds
            bonds = [(int(i), int(j)) for i,j in zip(left, right)]

            #we also add bonds at replication forks, to tie replicated and unreplicated strands together
            replication_bonds=[]
            if np.any(fork<0): #replication hasn't started yet. We tie the first two monomers to their replicates
                replication_bonds.append((0,self.N))
                replication_bonds.append((self.N-1, 2*self.N-1))
            else:
                behind_0=max(fork[0]-1,0)
                behind_1=min(fork[1]+1, self.N-1)
                replication_bonds.append((behind_0,behind_0+self.N)) #attach monomers behind fork
                replication_bonds.append((behind_1, behind_1+self.N)) #attach monomer behind fork

            allBonds.append(bonds+replication_bonds)
            smcs.append(bonds)
            forks.append(fork)

        #initialize the lists of bonds etc
        self.allBonds = allBonds
        self.smcs = smcs
        self.forkpos=forks
        self.uniqueBonds = list(set(sum(allBonds, [])))

        #adding forces and getting bond indices
        self.bondInds = []

        self.curBonds = allBonds.pop(0)
        self.curSmcs = smcs.pop(0)
        self.curFork = forks.pop(0)
        self.current_time=0

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

        self.current_time+=self.dt
        pastBonds = self.curBonds
        pastFork = self.curFork
        #Get parameters for next time step
        self.curBonds = self.allBonds.pop(0)
        self.curSmcs=self.smcs.pop(0)
        self.curFork=self.forkpos.pop(0)

        #Change bonds
        bondsRemove = [i for i in pastBonds if i not in self.curBonds]
        bondsAdd = [i for i in self.curBonds if i not in pastBonds]
        if verbose:
            print("{0} new bonds, {1} bonds removed".format(len(bondsAdd), len(bondsRemove)))
        bondsToChange = bondsAdd + bondsRemove
        bondsIsAdd = [True] * len(bondsAdd) + [False] * len(bondsRemove)

        for bond, isAdd in zip(bondsToChange, bondsIsAdd):
            if bond in self.bondToInd.keys():
                ind = self.bondToInd[bond] 
                paramset = self.activeParamDict if isAdd else self.inactiveParamDict
                self.bondForce.setBondParameters(ind, bond[0], bond[1], **paramset)  # actually updating bonds
            else:
                print("Key ", bond, "not found!")

        #Update the confinement size
        height=self.bacterium.t_to_height(self.current_time)

        context.setParameter("rounded_cap_cylindrical_confinement_top",height/2) #grow both sides by updating both
        context.setParameter("rounded_cap_cylindrical_confinement_bottom",-height/2)

        #update tethers...
        if self.num_tethers==2:
            old_ori, new_ori=self.bacterium.t_to_z_oris(self.current_time)
            self.tether.setParticleParameters(0,0,old_ori)
            self.tether.setParticleParameters(1,1,new_ori)
        elif self.num_tethers==1:
            old_ori, new_ori=self.bacterium.t_to_z_oris(self.current_time)
            self.tether.setParticleParameters(0,0,old_ori)

        if self.num_tethers>0:
            self.tether.updateParametersInContext(context)

        if np.any(self.curFork!=pastFork):
            #some sites were replicated...
            N=self.N

            #fetch positions, since we need to add the new monomers
            state=context.getState(getPositions=True, getVelocities=True)
            positions = state.getPositions(asNumpy=True)
            velocities = state.getVelocities(asNumpy=True)

            # Turn on excluded volume and springs for replicated monomers
            for i in range(max(0,pastFork[0]),self.curFork[0]):
                if i>0: #take the average position between the replicate and the monomer replicated before
                    positions[i+N]=np.mean(positions[[i,i-1+N]], axis=0)
                    velocities[i+N]=np.mean(velocities[[i,i-1+N]], axis=0)
                #else i==0; this monomer was tethered in right place

                self.excl.setParticleParameters(i+N,self.excl.getParticleParameters(i)) #turn on excluded volume

                if i>0: #turn on the spring behind the monomer
                    p1,p2,l,k=self.bondForce.getBondParameters(i-1)
                    assert (p1==i-1 and p2==i), f"Error: got particles {p1} and {p2} for bond {i-1}!"
                    self.bondForce.setBondParameters(N+i-1,p1+N,p2+N,l,k)

            for i in range(min(pastFork[1], N-1),self.curFork[1],-1):
                if i<N-1:
                    positions[i+N]=np.mean(positions[[i, i+1+N]], axis=0) #mean of replicate and the monomer replicated before
                    velocities[i+N]=np.mean(velocities[[i, i+1+N]], axis=0)
                #else i==N-1. this monomer was tetherd in place

                self.excl.setParticleParameters(i+N,self.excl.getParticleParameters(i))

                #Add spring behind the replicated monomer
                p1,p2,l,k=self.bondForce.getBondParameters(i)
                assert (p1==min(i,(i+1)%N) and p2==max(i,(i+1)%N)), f"Error: got particles {p1} and {p2} for bond {i+1}!"
                self.bondForce.setBondParameters(i+N,p1+N,p2+N,l,k)

            #Update the positions of the newly replicated monomers
            context.setPositions(positions)
            context.setVelocities(velocities)
            self.excl.updateParametersInContext(context)

        self.bondForce.updateParametersInContext(context)  # now run this to update things in the context

        return self.curSmcs, self.curFork #these can be saved for the configuration

def run_simulation(smcTrajFolder, bacterium, monomer_wig,\
        steps_per_sample, smcBondDist, smcBondWiggleDist, save_folder,\
        saveEveryConfigs, GPU_choice = 0, F_z=0., mass=100,\
        col_rate=0.1, trunc=0.5, top_monomer=0, num_tethers=0,\
        dt_1D=1/60):
    """
    Run simulations of chromosome in flat confinement with loop extrusion.

    Args:
        smcTrajFolder (str): folder with SMC trajectories
        bacterium (cell): bacterial species object
        monomer_wig (float): monomer wiggle distance
        steps_per_sample (int): number of 3D sim steps per sample
        smcBondDist (float): SMC bond distance
        smcBondWiggleDist (float): SMC bond wiggle distance
        save_folder (str): folder to save simulation data
        saveEveryConfigs (int): save every n configurations
        GPU_choice (int): GPU choice
        F_z (float): confinement force, also for tethers
        col_rate (float): collision rate
        trunc (float): excluded volume strength
        top_monomer (int): top monomer for start and tethering
        num_tethers (int): number of tethers
        dt_1D (float): 1D simulation time step in minutes
    Returns:
        None
    """
    smcSteps, numSims, trajectoryFiles=num_time_steps_trajectories(smcTrajFolder)
    N=bacterium.N
    polymerSteps= smcSteps*steps_per_sample

    savedSamples= smcSteps//saveEveryConfigs
    print(f"Got {smcSteps} steps and {numSims} sims.\nSaving {savedSamples} 3D configurations per trajectory.")

    block=0 #start count from zero   

    # clean up the simulation directory
    folder = save_folder
    if os.path.exists(folder):
        shutil.rmtree(folder)

    # create the reporter class
    reporter = HDF5Reporter(folder=folder, max_data_length=savedSamples) #one file per smc trajectory

    # Iterate over various BondUpdaterInitializations
    for BondUpdaterCount in range(numSims):
        smcFile=trajectoryFiles[BondUpdaterCount]

        # create the starting conformation
        start_data =bacterium.start_point_replicating(top_monomer) #unreplicated chromosome spread across cell

        #Now feed bond generators to BondUpdater 
        BondUpdater = simulationBondUpdater(bacterium, smcFile, trunc, num_tethers, dt_1D)

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
                PBCbox=False, #EDIT: no boundaries
                precision="mixed",
                verbose=False)  # timestep not necessary for variableLangevin

        a.set_data(start_data)  # loads polymer.

        # -----------Adding forces ---------------
        a.add_force(
            forcekits.polymer_chains(
                a,
                chains=[(0, N, True), (N, 2*N, True)], #two circular chromosomes
                bond_force_func=forces.harmonic_bonds, # adds harmonic bonds for polymers
                bond_force_kwargs={
                    'bondLength':1.0, # Bond length
                    'bondWiggleDistance':np.concatenate((monomer_wig*np.ones(N+1), np.zeros(N-2), [monomer_wig])), # Bond distance will fluctuate this much
                },

                angle_force_func=forces.angle_force,
                angle_force_kwargs={
                    'k':0.05 # we are making a very flexible polymer, basically not necessary here
                },
                nonbonded_force_func=forces.polynomial_repulsive_with_exclusions, # this is the excluded volume potential
                nonbonded_force_kwargs={
                    'trunc': np.concatenate((np.sqrt(trunc)*np.ones(N+1),np.zeros(N-2), [trunc])),
                    'radiusMult':1.05, # this is from old code
                },
                except_bonds=True,
            )
        )

        #Initialize confinement.
        a.add_force(forces.rounded_cap_cylindrical_confinement(a,bacterium.radius,bottom=-bacterium.L_0/2,k=10*F_z,top=bacterium.L_0/2))

        if num_tethers==2:
            z_ori, new_ori=bacterium.t_to_z_oris(0)
            a.add_force(forces.tether_particles(a,[top_monomer,top_monomer+N],k=[0,0,F_z],positions=[z_ori,z_ori])) #start with unreplicated; both oris below
        else:
            z_ori, new_ori=bacterium.t_to_z_oris(0)
            a.add_force(forces.tether_particles(a,[top_monomer],k=[0,0,F_z],positions=[z_ori])) #only one ori tethered

        # -----------Initialize bond updater. Add bonds ---------------
        a.step = block
        kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
        bondDist = smcBondDist * a.length_scale

        activeParams = {"length":bondDist,"k":kbond}
        inactiveParams = {"length":bondDist, "k":0}

        #Pass the forces and parameters to the BondUpdater
        BondUpdater.setParams(activeParams, inactiveParams)

        #Perform LEF simulation, sample bonds and fetch the first ones
        print("Load loop-extrusion data")
        if num_tethers==0:
            BondUpdater.LEF_simulation(a.force_dict['harmonic_bonds'],a.force_dict['rounded_cap_cylindrical_confinement'],
                a.force_dict['polynomial_repulsive_with_exclusions'],None, 
                smcTimeSteps=smcSteps) 
        else:
            BondUpdater.LEF_simulation(a.force_dict['harmonic_bonds'],a.force_dict['rounded_cap_cylindrical_confinement'],
                a.force_dict['polynomial_repulsive_with_exclusions'],a.force_dict['Tethers'], 
                smcTimeSteps=smcSteps) 

        # Minimize energy for first bonds
        print("Polymer burn-in")
        if BondUpdaterCount==0:
            a.local_energy_minimization() 
        else:
            a._apply_forces()

        smcs = BondUpdater.step(a.context) #get first bonds, update context
        #Docs say first steps after energy minimization have large error; don't save
        a.integrator.step(500*steps_per_sample)
        
        a.step=block
        if num_tethers==0:
            a.context.setParameter("Tethers_kz", 0) 

        # Iterate over simulation time steps within each BondUpdater 
        for i in tqdm(range(smcSteps-2)):
            smcs, fork = BondUpdater.step(a.context) #loop-extrusion step
            if i % saveEveryConfigs == 0: #3D polymer steps plus save 
                # save SMC, fork and monomer positions
                a.do_block(steps=steps_per_sample, save_extras={"SMCs":smcs, "fork":fork, "simulation_run":BondUpdaterCount}) 
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
