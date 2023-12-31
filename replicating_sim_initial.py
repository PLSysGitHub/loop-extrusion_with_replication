import argparse
import sys
import os
from itertools import product

def create_parser():
    parser = argparse.ArgumentParser(description='Run loop-extrusion simulations on a replicating chromosome.')
    parser.add_argument('GPU', type=int, nargs=1, help="Number for GPU to use")
    parser.add_argument("--no_smcs", action='store_true', help="Add flag if want no loop-extruders, ie just replication")
    parser.add_argument('--topological', action='store_true', help="Add flag if want topological loop-extruders")
    parser.add_argument('--num_tethers',type=int, choices=[0,1,2], default=0, help="0,1 or 2, sets how many oris are tethered")
    parser.add_argument('-M', '--num_condensins',type=int, default=40, help="Number of loop-extruders on single chromosome")
    parser.add_argument('-c', '--col_rate', default=0.03,type=float, help="Collision rate that sets drag force")
    parser.add_argument('-t', '--trunc', default=1.5,type=float, help="trunc parameter for strength of excluded volume")
    parser.add_argument('-f', '--fall_rate_fork', default=0.0,type=float, help="The off-loading rate of condensins at the replication fork")
    parser.add_argument('-p', '--pull-force', default=2.0,type=float, help="The force for the cylindrical confinement and ori-tether (if present)")
    parser.add_argument('-r', '--relative_fork_rate', default=1.488,type=float, help="The speed of a replication fork compared to a loop-extruder")
    parser.add_argument('-n', '--num_trajectories', default=100,type=int, help="The number of loop-extrusion trajectories to simulate")
    parser.add_argument('--top_monomer', default=0, type=int, help="The index of the monomer that is initialised at a pole, and possibly tethered")
    return parser
 
def get_kb_from_theory(num_extruders):
    d = 358 # kb
    v3 = 0.46
    v2 = 0.59
    v1 = 0.83
    n_encounters = 850*(0.4*num_extruders/1371) 
    tb = d/n_encounters/v3 - d/n_encounters/v2*(1-n_encounters/d)
    kb = 1/tb
    return kb

def out_dir_name_smcs(save_folder, GPU_num, stallFork, N, col_rate, separation, parSval, lifetime, fork_rate, stoch,steps_per_sample, trunc, pull_force):

    fname = f"GPU_{GPU_num}_stallFork_{stallFork}_N_{N}_colRate_{col_rate}_sep_{int(separation)}"+ \
            f"_parSstrength_{parSval}_lifetime_{int(lifetime)}_rateFork_{fork_rate}"+ \
            f"_stoch_{stoch}_sps_{steps_per_sample}_trunc_{trunc}_pullF_{pull_force}"
    return os.path.join(save_folder,fname)

def out_dir_name(save_folder, GPU_num, N, col_rate,fork_rate, stoch,steps_per_sample, trunc, pull_force):

    fname = f"GPU_{GPU_num}_N_{N}_colRate_{col_rate}_rateFork_{fork_rate}"+ \
            f"_stoch_{stoch}_sps_{steps_per_sample}_trunc_{trunc}_pullF_{pull_force}"
    return os.path.join(save_folder,fname)

def main():
    parser=create_parser()
    args = parser.parse_args()
    GPU=args.GPU[0]
    if args.no_smcs: 
        import loop_extrusion_replication.replicating_simulations.no_smcs as sim
        if args.num_tethers==0:
            save_folder = "Initial_No_smcs_no_tether"
            print("Running simulations with no tether and no loop-extrusion")
        elif args.num_tethers==2:
            save_folder = "Initial_No_smcs"
            print("Running simulations with ori-tether and no loop-extruders")
        else:
            save_folder = "Initial_No_smcs_one_tether"
    elif args.topological:
        import loop_extrusion_replication.replicating_simulations.topological as sim
        if args.num_tethers==0:
            save_folder = "Initial_Topological_smcs_no_tether"
            print("Running simulations with no tether and topological loop-extruders")
        elif args.num_tethers==2:
            save_folder = "Initial_Topological_smcs"
            print("Running simulations with ori-tether and topological loop-extruders")
        else:
            save_folder = "Initial_Topological_smcs_one_tether"
            print("Running simulations with one tether and topological loop-extruders")

    else:#nontopological
        import loop_extrusion_replication.replicating_simulations.nontopological as sim
        if args.num_tethers==0:
            save_folder = "Initial_Nontopological_smcs_no_tether"
            print("Running simulations with no tether and nontopological loop-extruders")
        elif args.num_tethers==2:
            save_folder = "Initial_Nontopological_smcs"
            print("Running simulations with ori-tether and nontopological loop-extruders")
        else:
            save_folder= "Initial_Nontopological_smcs_one_tether"
            print("Running simulations with one tether and nontopological loop-extruders")
            
    #Set rest of parameters
    
    #Dimensions
    N=404 #number of 3D monomers per chromosome
    if args.no_smcs:
        N_1D=N #without smcs, 1D and 3D dimensions are the same
    else:
        N_1D = 4040 #keep same 1D dimensions as used by Brandao et.al. 2021; collisions as frequent

    monomer_size=129 #nm, Messelink et.al. 2021
    monomer_wig=17**0.5/monomer_size/(N/404) #var for distance between 10 kb is 17 nm, divide by beads/10 kb
    smcBondDist = 50/monomer_size #Condensin roughly 50 nm large
    smcBondWiggleDist = 0.2*smcBondDist
    
    #Confinement
    radius= 750/2/monomer_size #in sim units. Width 750 as Messelinck et.al. 2021
    height=2252/monomer_size #Messelink et.al. 2021
    z_ori=height/2-343/monomer_size #where oris are tethered
    
    #1D simulation rates
    base_stochasticity= 0.05 #loop extruder step rate per simulation step
    fork_rate=base_stochasticity*args.relative_fork_rate #rate fork from Hi-C/smcs in caulobacter Tran 2017

    if not args.no_smcs:
        separation=N_1D//args.num_condensins
        #Loop extruder rates; Brandao et.al. 2021
        knockOffProb = 1
        k_bypass = get_kb_from_theory(args.num_condensins)
        k_knock = k_bypass/20
        kOriToTer_kTerToOri_kBypass=[k_knock,k_knock,k_bypass]

        #Loading sites and offloading sites
        parSsites = [0]
        wind = 0 
        terSites = [N_1D*195//405,N_1D*205//405]
        terSiteStrength = 0.05*base_stochasticity # rate of dissociation (1/(simulation time step))
        parSval=N_1D #50% loaded at ori
        parSstrengths=[parSval/len(parSsites)]*len(parSsites)

    life_time = N_1D/2/base_stochasticity # time in (simulation time) step units #C crescentus: level off around 600 kb
    translocator_initialization_steps = round(life_time*50) # for SMC translocator

    # Simulations give samples_per_trajectory timepoints
    # each time point has saveEveryConfigs*smcStepsPerBlock loop extrusion simulation timepoints
    # Overall, num_trajectories simulations are done

    smcStepsPerBlock = int(N_1D/4040/base_stochasticity) #for ten times smaller 1D simulations need ten times less steps

    # saving for polymer simulation
    steps_per_sample = 2500 #polymer simulation steps for each block of smc positions
    saveEveryConfigs = 1

    restartConfigurationEvery = round(0.15*N_1D/(2*fork_rate*smcStepsPerBlock)) #stop simulations early on; we're interested in short time scales
    restartConfigurationEvery -= restartConfigurationEvery%saveEveryConfigs #just to make restartConfigurationEvery divisible by saveEveryConfigs
    samples_per_trajectory=int(restartConfigurationEvery/saveEveryConfigs)

    savedSamples = args.num_trajectories*samples_per_trajectory


    #Output directory creation
    if args.no_smcs:
        out_dir=out_dir_name(save_folder, GPU, N, args.col_rate,fork_rate, base_stochasticity, steps_per_sample, args.trunc, args.pull_force)      
    else:
        out_dir=out_dir_name_smcs(save_folder, GPU, args.fall_rate_fork, N, args.col_rate,separation, parSstrengths[0], life_time, fork_rate, base_stochasticity, steps_per_sample, args.trunc, args.pull_force)      
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    

    #Actually start the simulations
    if args.no_smcs:
        print("Start sims no smcs")
        sim.run_simulation(monomer_size,monomer_wig, N, N_1D, steps_per_sample, radius, height,z_ori, smcStepsPerBlock, \
            smcBondDist, smcBondWiggleDist, out_dir, saveEveryConfigs, \
            savedSamples, restartConfigurationEvery,GPU_choice=GPU, F_z=args.pull_force,col_rate=args.col_rate,\
            trunc=args.trunc, fork_rate=fork_rate, top_monomer=args.top_monomer, num_tethers=args.num_tethers)
    else:
        print("Start sims with smcs")
        sim.run_simulation(monomer_size,monomer_wig,knockOffProb, kOriToTer_kTerToOri_kBypass, parSsites,\
                wind, terSites, terSiteStrength, parSstrengths,\
                separation,life_time, N,N_1D, base_stochasticity, translocator_initialization_steps, \
                steps_per_sample, radius, height,z_ori, smcStepsPerBlock, \
                smcBondDist, smcBondWiggleDist, out_dir, saveEveryConfigs, \
                savedSamples, restartConfigurationEvery,\
                GPU_choice=GPU, F_z=args.pull_force, col_rate=args.col_rate,\
                trunc=args.trunc, fork_rate=fork_rate, stallFork=args.fall_rate_fork, top_monomer=args.top_monomer,num_tethers=args.num_tethers)

main()
