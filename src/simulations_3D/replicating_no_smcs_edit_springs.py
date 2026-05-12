import pickle
from tqdm import tqdm
import os
import time
import numpy as np
import polychrom
import simtk.unit
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

from ..bacterial_species import cell

class simulationBondUpdater(object):
    """
    This class precomputes simulation bonds for faster dynamic allocation. 
    """

    def __init__(self,bacterium,fork_rate, num_tethers, k0, decay_l_behind=0, factor_behind=1, decay_l_ahead=0, factor_ahead=1,start_compression_t=5, dt=1/60, tie_forks=False):
        """
        :param bacterium: a cell object
        :param fork_rate: the rate of fork movement
        :param num_tethers: the number of tethers
        :param k0: the spring constant
        :param decay_l_behind: the decay length behind the forks
        :param factor_behind: the factor of spring extension/compaction behind the forks
        :param decay_l_ahead: the decay length ahead of the forks
        :param factor_ahead: the factor of spring extension/compaction ahead of the forks
        :param dt: the time step in minutes
        :param tie_forks: if True, tie the forks
        """
        self.bacterium=bacterium
        self.N=bacterium.N
        self.fork_rate=fork_rate
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
        self.tie_forks=tie_forks
        self.current_time=0
        self.dt=dt
        self.is_replicated=np.concatenate((np.ones(self.N, dtype=bool),np.zeros(self.N, dtype=bool)))

    def setParams(self, activeParamDict, inactiveParamDict):
        """
        A method to set parameters for bonds.
        It is a separate method because you may want to have a Simulation object already existing

        :param activeParamDict: a dict (argument:value) of addBond arguments for active bonds
        :param inactiveParamDict:  a dict (argument:value) of addBond arguments for inactive bonds
        """
        self.activeParamDict = activeParamDict #pre-replication bond
        self.inactiveParamDict = inactiveParamDict #pre-replication bond

    def LEF_simulation(self, bondForce, cylinder, tether, excl, smcSteps):
        """
        A method that milks smcTranslocator object
        and creates a set of unique bonds, etc.

        :param bondForce: a bondforce object (new after simulation restart!)
        :param cylinder: the cylindrical confinement force object
        :param tether: tether force object that holds oris in place
        :param excl: excluded volume force object
        :param smcSteps: number of smcTranslocator steps per block
        :return:
        """
        if len(self.allBonds) != 0:
            raise ValueError("Not all bonds were used; {0} sets left".format(len(self.allBonds)))


        self.bondForce = bondForce
        self.cylinder = cylinder
        self.tether = tether
        self.excl = excl

        forks=[]
        allBonds = []
        f1=-1
        f2=self.N

        for dummy in range(smcSteps):
            if f1<self.N/2 and np.random.rand()<self.fork_rate:
                f1+=1
            if f2>self.N/2 and np.random.rand()<self.fork_rate:
                f2-=1

            forks.append([f1,f2])
            replication_bonds=[] #bonds that keep linear strand attached
            if f1>0:
                replication_bonds.append((f1-1,f1-1+self.N)) #attach two replicates behind fork
            else:
                replication_bonds.append((0,self.N)) #before replication starts, attach ori copies

            if f2<self.N-1:
                replication_bonds.append((f2+1, f2+1+self.N)) #attach two replicates behind fork
            else:
                replication_bonds.append((self.N-1, 2*self.N-1)) #attach replicates of N-1 (next to ori)

            if self.tie_forks:
                replication_bonds.append((f1,f2))

            allBonds.append(replication_bonds)

        self.allBonds = allBonds
        self.forkpos=forks
        self.uniqueBonds = list(set(sum(allBonds, [])))

        #adding forces and getting bond indices
        self.bondInds = []

        self.curBonds = allBonds.pop(0)
        self.curFork = forks.pop(0)
        self.current_time=0

        for bond in self.uniqueBonds:
            paramset = self.activeParamDict if (bond in self.curBonds) else self.inactiveParamDict #create all bonds; some inactive
            ind = self.bondForce.addBond(bond[0], bond[1], **paramset)
            self.bondInds.append(ind) #the index for the bond

        self.bondToInd = {i:j for i,j in zip(self.uniqueBonds, self.bondInds)} #map bond monomers to bond index

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

        self.prevFork = self.curFork
        pastBonds=self.curBonds
        self.curFork=self.forkpos.pop(0)
        self.current_time+=self.dt

        #Get parameters for next time step
        self.curBonds = self.allBonds.pop(0)

        #Change bonds
        bondsRemove = [i for i in pastBonds if i not in self.curBonds]
        bondsAdd = [i for i in self.curBonds if i not in pastBonds]
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
            p0, params0 = self.tether.getParticleParameters(0)
            p1, params1 = self.tether.getParticleParameters(1)
            self.tether.setParticleParameters(0,p0,old_ori)
            self.tether.setParticleParameters(1,p1,new_ori)
        elif self.num_tethers==1:
            old_ori, new_ori=self.bacterium.t_to_z_oris(self.current_time)
            p0, params0 = self.tether.getParticleParameters(0)
            self.tether.setParticleParameters(0,p0,old_ori)

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
            for i in range(max(0,self.prevFork[0]),self.curFork[0]):
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

            for i in range(min(self.prevFork[1], N-1),self.curFork[1],-1):
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

        assert (np.all(self.is_replicated[:self.N]==True) and np.all(self.is_replicated[self.N:self.curFork[0]+self.N]==True) and np.all(self.is_replicated[self.curFork[1]+1+self.N:2*self.N]==True)), f"Error! self.is_replicated inconsistent with curFork. curFork={self.curFork}"

        self.bondForce.updateParametersInContext(context)  # now run this to update things in the context
        return self.curFork #these can be saved for the configuration


def run_simulation(bacterium, monomer_wig, smcBondDist, smcBondWiggleDist, \
        steps_per_sample, save_folder, smcSteps, numSims,\
        delta_t_sec, saveEveryConfigs,GPU_choice = 0, F_z=0.,\
        col_rate=0.1, trunc=0.5, top_monomer=0, num_tethers=0,\
        no_confinement=False,infinite_tube=False, mass=100,\
        decay_l_behind=0, factor_behind=1, decay_l_ahead=0,\
        factor_ahead=1, compression_t=5, platform="cuda"):
    """
    This function initiates the bacterial chromosome simulation.

    Parameters:
    bacterium : cell object
        The bacterial chromosome object to be simulated.
    monomer_wig : float
        The wiggle distance for monomers.
    smcBondDist : float
        The distance for SMC bond.
    smcBondWiggleDist : float
        The wiggle distance for SMC bond.
    steps_per_sample : int
        Number of steps per sample.
    save_folder : str
        The folder where simulation results will be saved.
    smcSteps : int
        Number of SMC steps.
    numSims : int
        Number of simulations to run.
    delta_t_sec : float
        Time step in seconds.
    saveEveryConfigs : int
        Save configuration every specified number of steps.
    GPU_choice : int, optional
        GPU choice for running the simulation (default is 0).
    F_z : float, optional
        Force for confinement strength (default is 0).
    col_rate : float, optional
        Collision rate (default is 0.1).
    trunc : float, optional
        Sets strength of excluded volume interactions in k_B T (default is 0.5).
    top_monomer : int, optional
        Index of the top monomer for start configuration and tethering forces (default is 0).
    num_tethers : int, optional
        Number of tethers (default is 0).
    no_confinement : bool, optional
        If True, no confinement is applied (default is False).
    infinite_tube : bool, optional
        If True, an infinite tube is used (default is False).
    decay_l_behind : float, optional
        Decay length behind forks (default is 0).
    factor_behind : float, optional
        Factor of spring extension/compaction behind forks (default is 1).
    decay_l_ahead : float, optional
        Decay length ahead of forks (default is 0).
    factor_ahead : float, optional
        Factor of spring extension/compaction ahead of forks (default is 1).

    Returns:
    None
    """
    fork_rate=bacterium.rateReplication/60*delta_t_sec #probability of a step per 1D sim step

    N=bacterium.N
    polymerSteps= smcSteps*steps_per_sample

    savedSamples= smcSteps//saveEveryConfigs
    print(f"Simulating a polymer of {N} monomers")
    pulled=top_monomer+N
    block=0 #start count from zero   

    # clean up the simulation directory
    folder = save_folder
    if os.path.exists(folder):
        shutil.rmtree(folder)

    # create the reporter class
    reporter = HDF5Reporter(folder=folder, max_data_length=150) 

    # Iterate over various BondUpdaterInitializations
    for BondUpdaterCount in range(numSims):
        start_data = bacterium.start_point_replicating(top_monomer)

        # simulation parameters are defined below 
        a = Simulation(
                platform=platform,
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

        #Initialize confinement.
        if infinite_tube:
            a.add_force(forces.rounded_cap_cylindrical_confinement(a,bacterium.radius,bottom=None,k=10*F_z))
        elif not no_confinement:
            a.add_force(forces.rounded_cap_cylindrical_confinement(a,bacterium.radius,bottom=-bacterium.L_0/2,k=10*F_z,top=bacterium.L_0/2))

        if num_tethers==2:
            z_ori, new_ori=bacterium.t_to_z_oris(0)
            a.add_force(forces.tether_particles(a,[top_monomer,pulled],k=[0,0,F_z],positions=[z_ori,z_ori])) #start with unreplicated; both oris below
        else:
            z_ori, new_ori=bacterium.t_to_z_oris(0)
            a.add_force(forces.tether_particles(a,[top_monomer],k=[0,0,F_z],positions=[z_ori])) #only one ori tethered
        
        # -----------Initialize bond updater. Add bonds ---------------
        a.step = block
        k0 = a.kbondScalingFactor / (monomer_wig ** 2)
        kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
        bondDist = smcBondDist * a.length_scale
        activeParams = {"length":bondDist,"k":kbond}
        inactiveParams = {"length":bondDist, "k":0}

        #Perform LEF simulation, sample bonds and fetch the first ones
        print("Fork position simulations")
        
        BondUpdater = simulationBondUpdater(bacterium, fork_rate, num_tethers, k0, decay_l_behind, factor_behind, decay_l_ahead, factor_ahead, compression_t)

        #Pass the forces and parameters to the BondUpdater
        BondUpdater.setParams(activeParams, inactiveParams)

        if num_tethers==0:
            BondUpdater.LEF_simulation(a.force_dict['harmonic_bonds'],a.force_dict['rounded_cap_cylindrical_confinement'],
                None, a.force_dict['polynomial_repulsive_vary_trunc_r'],
                smcSteps=smcSteps) 
        else:
            BondUpdater.LEF_simulation(a.force_dict['harmonic_bonds'],a.force_dict['rounded_cap_cylindrical_confinement'],
                a.force_dict['Tethers'], a.force_dict['polynomial_repulsive_vary_trunc_r'],
                smcSteps=smcSteps) 

        # Minimize energy for first bonds
        print("Polymer burn-in")
        if BondUpdaterCount==0:
            a.local_energy_minimization() 
        else:
            a._apply_forces()        
            
        fork = BondUpdater.step(a.context) #get first bonds, update context
        
        #Docs say first steps after energy minimization have large error; don't save
        a.integrator.step(500*steps_per_sample)
        a.step=block
        if num_tethers==0:
            a.context.setParameter("Tethers_kz", 0) 

        # Iterate over simulation time steps within each BondUpdater 
        for i in tqdm(range(smcSteps-2)):
            # BondUpdater updates bonds at each step.
            fork = BondUpdater.step(a.context)
            if i % saveEveryConfigs == 0: #3D polymer steps plus save 
                a.do_block(steps=steps_per_sample, save_extras={"fork":fork, "simulation_run":BondUpdaterCount}) 
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
