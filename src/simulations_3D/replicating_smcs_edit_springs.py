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
    files=os.listdir(folder_name)
    filtered_files = [os.path.join(folder_name,f) for f in files if "simulation_" in f]
    
    with h5py.File(filtered_files[0], 'r') as f:
        ts=f['ts'][:]

    nt=ts.size
    nsims=len(filtered_files)

    return nt, nsims, filtered_files

def load_smcs_forks_from_h5(filename, t):

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

    def __init__(self,bacterium, smcTrajectoryFile, trunc, num_tethers, k0, decay_l_behind=0, factor_behind=1, decay_l_ahead=0, factor_ahead=1, start_compression_t=5,dt=1/60):
        """
        :param bacterium: a cell type object
        :param smcTrajectoryFile: a file with SMC trajectories
        :param trunc: excluded volume strength
        :param num_tethers: number of tethers
        :param k0: spring constant
        :param decay_l_behind: decay length behind replication fork
        :param factor_behind: factor by which springs extend/contract behind replication fork
        :param decay_l_ahead: decay length ahead of replication fork
        :param factor_ahead: factor by which springs extend/contract ahead of replication fork
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
        self.decay_l_behind=decay_l_behind
        self.decay_l_ahead=decay_l_ahead
        self.factor_behind=factor_behind
        self.factor_ahead=factor_ahead
        self.start_compression_t=start_compression_t
        self.k0=k0
        self.dt=dt
        self.current_time=0
        self.is_replicated=np.concatenate((np.ones(self.N, dtype=bool),np.zeros(self.N, dtype=bool)))

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
        :param smcTimeSteps: number of time steps in the SMC trajectory file
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
            if fork[0]>0:
                replication_bonds.append((fork[0]-1,fork[0]-1+self.N))
            else:
                replication_bonds.append((0,self.N))

            if fork[1]<self.N-1 and fork[1]>-1:
                replication_bonds.append((fork[1]+1, fork[1]+1+self.N))
            else:
                replication_bonds.append((self.N-1, 2*self.N-1))

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
        self.prevFork= [0,self.N-1]
        self.current_time=0

        for bond in self.uniqueBonds:
            paramset = self.activeParamDict if (bond in self.curBonds) else self.inactiveParamDict
            ind = self.bondForce.addBond(bond[0], bond[1], **paramset)
            self.bondInds.append(ind)

        self.bondToInd = {i:j for i,j in zip(self.uniqueBonds, self.bondInds)}

        return self.curBonds

    def current_spring_factor(self, i):
        """
        Calculate the current spring factor for a given bond
        :param i: the bond index
        :return: the current spring factor
        """
        if i%self.N<self.curFork[0] or i%self.N>self.curFork[1]: #replicated, behind forks
            if self.decay_l_behind<=1:
                return 1
            else:
                d_fork=max(self.curFork[0]-i%self.N, i%self.N-self.curFork[1]) 
                return 1+(self.factor_behind-1)*np.exp(-d_fork/self.decay_l_behind)
        else: #not replicated; ahead of forks
            if self.decay_l_ahead<=1 or self.current_time<self.start_compression_t:
                return 1
            else:
                d_fork=min(self.curFork[1]-i%self.N, i%self.N-self.curFork[0])
                return 1+(self.factor_ahead-1)*np.exp(-d_fork/self.decay_l_ahead)

    def update_spring(self,i):
        """
        Update the spring length and strength for a given bond
        :param i: the bond index
        """
        if self.is_replicated[i]:
            current_factor=self.current_spring_factor(i)
            p1,p2,l,k=self.bondForce.getBondParameters(i)
            if self.is_replicated[p1] and self.is_replicated[p2]:
                k_new=self.k0*(1/current_factor)**2
                self.bondForce.setBondParameters(i,p1,p2,current_factor,k_new)

            #also update excluded volume radius
            old_params=self.excl.getParticleParameters(i) #strength and radius for excl volume of interacting copy
            self.excl.setParticleParameters(i, [old_params[0], 1.05*current_factor]) #adjust the radius

    def update_springs(self):
        """
        Update the spring lengths on the entire polymer
        """            

        #first chromosome copy
        for i in range(self.N):
            self.update_spring(i)

        #second chromosome copy; only replicated
        for i in range(self.prevFork[0]):
            self.update_spring(i+self.N)

        for i in range(self.prevFork[1]+1,self.N):
            self.update_spring(i+self.N)

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
        self.curFork=self.forkpos.pop(0)
        self.current_time+=self.dt

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

        #and tethers...
        if self.num_tethers==2:
            old_ori, new_ori=self.bacterium.t_to_z_oris(self.current_time)
            self.tether.setParticleParameters(0,0,old_ori)
            self.tether.setParticleParameters(1,1,new_ori)
        elif self.num_tethers==1:
            old_ori, new_ori=self.bacterium.t_to_z_oris(self.current_time)
            self.tether.setParticleParameters(0,0,old_ori)

        if self.num_tethers>0:
            self.tether.updateParametersInContext(context)

        if np.any(self.curFork!=self.prevFork):
            #some sites were replicated...

            self.update_springs() #recalculate spring lengths

            N=self.N

            #fetch positions, since we need to add the new monomers
            state=context.getState(getPositions=True, getVelocities=True)
            positions = state.getPositions(asNumpy=True)
            velocities = state.getVelocities(asNumpy=True)

            # Turn on excluded volume and springs for replicated monomers
            for i in range(self.prevFork[0],self.curFork[0]):
                if i>0: #take the average position between the replicate and the monomer replicated before
                    new_pos=np.mean(positions[[i,i-1+N]], axis=0)
                    positions[i+N]=new_pos
                    velocities[i+N]=np.mean(velocities[[i,i-1+N]], axis=0)
                #else i==0; this monomer is tethered to ori 1 before replication, and doesn't need to have position initialized

                self.excl.setParticleParameters(i+N,self.excl.getParticleParameters(i)) #turn on excluded volume

                if i>0: #turn on the spring behind the monomer
                    p1,p2,l,k=self.bondForce.getBondParameters(i-1)
                    assert (p1==i-1 and p2==i), f"Error: got particles {p1} and {p2} for bond {i-1}!"
                    self.bondForce.setBondParameters(N+i-1,p1+N,p2+N,l,k)

                self.is_replicated[i+N]=True

            for i in range(self.prevFork[1],self.curFork[1],-1):
                if i<N-1:
                    new_pos=np.mean(positions[[i, i+1+N]], axis=0) #mean of replicate and the monomer replicated before
                    positions[i+N]=new_pos
                    velocities[i+N]=np.mean(velocities[[i, i+1+N]], axis=0)
                #else i==N-1. Monomer tethered to replicate before replication; no need to initialize position

                self.excl.setParticleParameters(i+N,self.excl.getParticleParameters(i))

                #Add spring behind the replicated monomer
                p1,p2,l,k=self.bondForce.getBondParameters(i)
                assert (p1==min(i,(i+1)%N) and p2==max(i,(i+1)%N)), f"Error: got particles {p1} and {p2} for bond {i+1}!"
                self.bondForce.setBondParameters(i+N,p1+N,p2+N,l,k)
                self.is_replicated[i+N]=True

            #Update the positions of the newly replicated monomers
            context.setPositions(positions)
            context.setVelocities(velocities)
            self.excl.updateParametersInContext(context)

        assert np.all(self.is_replicated[:self.N]==True), "Unreplicated between 0:self.N!"
        assert np.all(self.is_replicated[self.N:self.curFork[0]+self.N]==True), f"Unreplicated between N and curFork[0]+N, curFork={self.curFork}, {self.is_replicated[self.N:self.curFork[0]+self.N]}"
        assert np.all(self.is_replicated[self.curFork[1]+1+self.N:2*self.N]==True), f"Error! unreplicated between curFork[1]+N and 2*N curFork={self.curFork}"

        self.bondForce.updateParametersInContext(context)  # now run this to update things in the context
        self.prevFork = self.curFork

        return self.curSmcs, self.curFork #these can be saved for the configuration

def run_simulation(smcTrajFolder, bacterium, monomer_wig,\
        steps_per_sample, smcBondDist, smcBondWiggleDist, save_folder,\
        saveEveryConfigs, GPU_choice = 0, F_z=0., mass=100,\
        col_rate=0.1, trunc=0.5, top_monomer=0, num_tethers=0,\
        decay_l_behind=0, factor_behind=1, decay_l_ahead=0, factor_ahead=1,\
        compression_t=5, dt_1D=1/60):

    """
    Run simulations of chromosome in flat confinement with loop extrusion.

    Args:
        smcTrajFolder: str - folder with SMC trajectories
        bacterium: cell - bacterial species object
        monomer_wig: float - monomer wiggle distance
        steps_per_sample: int - number of 3D simulation steps per sample
        smcBondDist: float - SMC bond distance
        smcBondWiggleDist: float - SMC bond wiggle distance
        save_folder: str - folder to save simulation data
        saveEveryConfigs: int - number of steps between saving configurations
        GPU_choice: int - GPU choice
        F_z: float - force in z direction for tether forces, as well as for confinement
        col_rate: float - collision rate
        trunc: float - sets excluded volume strength
        top_monomer: int - index of top monomer for start configuration and tethering
        num_tethers: int - number of tethers
        decay_l_behind: float - decay length behind replication fork
        factor_behind: float - factor by which springs extend/contract behind replication fork
        decay_l_ahead: float - decay length ahead of replication fork
        factor_ahead: float - factor by which springs extend/contract ahead of replication fork
        dt_1D: float - time step in minutes for 1D simulations
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
                nonbonded_force_func=forces.polynomial_repulsive_vary_trunc_r, # this is the excluded volume potential
                nonbonded_force_kwargs={
                    'trunc': np.concatenate((np.sqrt(trunc)*np.ones(N+1),np.zeros(N-2), [trunc])),
                    'radiusMult':1.05, # this is from old code
                    'maxRadius' : np.max([1,factor_behind, factor_ahead]),
                },
                except_bonds=True,
            )
        )

        #Initialize confinement. During explosion, the confinement will be removed.
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
        k0 = a.kbondScalingFactor / (monomer_wig ** 2)

        activeParams = {"length":bondDist,"k":kbond}
        inactiveParams = {"length":bondDist, "k":0}

        #Now feed bond generators to BondUpdater 
        BondUpdater = simulationBondUpdater(bacterium, smcFile, trunc, num_tethers, k0, decay_l_behind, factor_behind, decay_l_ahead, factor_ahead, compression_t, dt_1D)

        #Pass the forces and parameters to the BondUpdater
        BondUpdater.setParams(activeParams, inactiveParams)

        #Perform LEF simulation, sample bonds and fetch the first ones
        print("Load loop-extrusion data")
        if num_tethers==0:
            BondUpdater.LEF_simulation(a.force_dict['harmonic_bonds'],a.force_dict['rounded_cap_cylindrical_confinement'],
                a.force_dict['polynomial_repulsive_vary_trunc_r'],None, 
                smcTimeSteps=smcSteps) 
        else:
            BondUpdater.LEF_simulation(a.force_dict['harmonic_bonds'],a.force_dict['rounded_cap_cylindrical_confinement'],
                a.force_dict['polynomial_repulsive_vary_trunc_r'],a.force_dict['Tethers'], 
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
