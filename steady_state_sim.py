import argparse
import sys
import os
from itertools import product

def create_parser():
    parser = argparse.ArgumentParser(description='Run loop-extrusion simulations on a replicating chromosome.')
    parser.add_argument('GPU', type=int, nargs=1, help="Number for GPU to use")
    parser.add_argument("--no_smcs", action='store_true', help="Add flag if want no loop-extruders, ie just replication")
    parser.add_argument('--topological', action='store_true', help="Add flag if want topological loop-extruders")
    parser.add_argument('--forks_bound', action='store_true', help="Add flag if replication forks should be tied. Only without LEs.")
    parser.add_argument('--num_tethers',type=int, choices=[0,1,2], default=0, help="0,1 or 2, sets how many oris are tethered")
    parser.add_argument('--frac_travelled', default=0.5,type=float, help="Sets lifetime as fraction*monomers/speed. Default 0.5; ori to ter")
    parser.add_argument('--parS_strength', default=4040.,type=float, help="Loading preference at ori. Default leads to 50% chance of ori loading.")
    parser.add_argument('-M', '--num_condensins',type=int, default=40, help="Number of loop-extruders on single chromosome")
    parser.add_argument('--top_monomer',type=int, default=0, help="The monomer that will be at the pole for an initial unsegregated configuration")
    parser.add_argument('-c', '--col_rate', default=0.03,type=float, help="Collision rate that sets drag force")
    parser.add_argument('-t', '--trunc', default=1.5,type=float, help="trunc parameter for strength of excluded volume")
    parser.add_argument('-f', '--fall_rate_fork', default=0.0,type=float, help="The off-loading rate of condensins at the replication fork")
    parser.add_argument('-p', '--pull-force', default=2.0,type=float, help="The force for the cylindrical confinement and ori-tether (if present)")
    parser.add_argument('-R', '--replicated_length',type=int, required=True, help="The extent of replication, integer less than N")
    parser.add_argument('--segregated', action='store_true', help="Add flag for a starting conformation that's segregated")
    parser.add_argument('-n', '--num_trajectories', default=100,type=int, help="The number of loop-extrusion trajectories to simulate")
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

def out_dir_name_smcs(save_folder, GPU_num, R, stallFork, N, col_rate, separation, parSval, lifetime, stoch,steps_per_sample, trunc, pull_force):

    fname = f"GPU_{GPU_num}_R_{R}_stallFork_{stallFork}_N_{N}_colRate_{col_rate}_sep_{int(separation)}"+ \
            f"_parSstrength_{parSval}_lifetime_{int(lifetime)}"+ \
            f"_stoch_{stoch}_sps_{steps_per_sample}_trunc_{trunc}_pullF_{pull_force}"
    return os.path.join(save_folder,fname)

def out_dir_name(save_folder, GPU_num,R, N, col_rate, stoch,steps_per_sample, trunc, pull_force, top_monomer, start_segr):
    if not start_segr:
        fname = f"GPU_{GPU_num}_R_{R}_N_{N}_colRate_{col_rate}"+ \
            f"_stoch_{stoch}_sps_{steps_per_sample}_trunc_{trunc}_pullF_{pull_force}_top_mon_{top_monomer}"
    else:
        fname = f"GPU_{GPU_num}_R_{R}_N_{N}_SEGREGATED_colRate_{col_rate}"+ \
            f"_stoch_{stoch}_sps_{steps_per_sample}_trunc_{trunc}_pullF_{pull_force}_top_mon_{top_monomer}"
    return os.path.join(save_folder,fname)

def main():
    parser=create_parser()
    args = parser.parse_args()
    GPU=args.GPU[0]
    if args.no_smcs: 
        import loop_extrusion_replication.steady_state_simulations.no_smcs as sim
        if args.num_tethers==0:
            save_folder = "Steady_state_No_smcs_no_tether"
            print("Running simulations with no tether and no loop-extrusion")
        elif args.num_tethers==2:
            save_folder = "Steady_state_No_smcs"
            print("Running simulations with two tethers and no loop-extruders")
        else:
            print("Running simulations with one tether and no loop-extruders")
            save_folder = "Steady_state_No_smcs_one_tether"
        if args.forks_bound:
            save_folder=save_folder+"_tied_forks"
    elif args.topological:
        import loop_extrusion_replication.steady_state_simulations.topological as sim
        if args.num_tethers==0:
            save_folder = "Steady_state_Topological_smcs_no_tether"
            print("Running simulations with no tether and topological loop-extruders")
        elif args.num_tethers==2:
            save_folder = "Steady_state_Topological_smcs"
            print("Running simulations with ori-tether and topological loop-extruders")
        else:
            save_folder = "Steady_state_Topological_smcs_one_tether"
            print("Running simulations with one tether and topological loop-extruders")
        if args.forks_bound:
            print("Warning: no forks tied with loop-extruders!")

    else:#nontopological
        import loop_extrusion_replication.steady_state_simulations.nontopological as sim
        if args.num_tethers==0:
            save_folder = "Steady_state_Nontopological_smcs_no_tether"
            print("Running simulations with no tether and nontopological loop-extruders")
        elif args.num_tethers==2:
            save_folder = "Steady_state_Nontopological_smcs"
            print("Running simulations with ori-tether and nontopological loop-extruders")
        else:
            save_folder= "Steady_state_Nontopological_smcs_one_tether"
            print("Running simulations with one tether and nontopological loop-extruders")
        if args.forks_bound:
            print("Warning: no forks tied with loop-extruders!")
            
    #Set rest of parameters
    
    #Dimensions
    N=404 #number of 3D monomers per chromosome
    if args.no_smcs:
        N_1D=N #without smcs, this is just lattice for forks
    else:
        N_1D = 4040 #as in Brandao et al. 2021, keeping LE density same in 1D

    monomer_size=129#nm, Messelink et.al. 2021
    monomer_wig=17**0.5/monomer_size/(N/404) #var for distance between 10 kb is 17 nm, divide by beads/10 kb
    smcBondDist = 50/monomer_size #Condensin roughly 50 nm large
    smcBondWiggleDist = 0.2*smcBondDist
    
    #Confinement
    radius= 750/2/monomer_size #in sim units. Messelink et.al. 2021
    
    #1D simulation rates
    base_stochasticity= 0.05 #loop extruder step rate per simulation step

    if not args.no_smcs:
        separation=N_1D//args.num_condensins
        #Loop extruder rates
        knockOffProb = 1
        k_bypass = get_kb_from_theory(args.num_condensins)
        k_knock = k_bypass/20
        kOriToTer_kTerToOri_kBypass=[k_knock,k_knock,k_bypass]

        #Loading sites and offloading sites
        parSsites = [0]
        wind = 0 
        terSites = [N_1D*195//404,N_1D*205//404]
        terSiteStrength = 0.05*base_stochasticity # rate of dissociation (1/(simulation time step))
        parSval=args.parS_strength
        parSstrengths=[parSval/len(parSsites)]*len(parSsites)
        life_time = args.frac_travelled*N_1D/base_stochasticity # time in (simulation time) step units #C crescentus: level off around 600 kb
        translocator_initialization_steps = 100000 # for SMC translocator

    # Simulations give samples_per_trajectory timepoints
    # each time point has saveEveryConfigs*smcStepsPerBlock loop extrusion simulation timepoints
    # Overall, num_trajectories simulations are done

    smcStepsPerBlock = int(N_1D/4040/base_stochasticity) #for ten times smaller 1D simulations need ten times less steps

    # saving for polymer simulation
    steps_per_sample = 2500 #polymer simulation steps for SMC update
    saveEveryConfigs = 200

    restartConfigurationEvery =150000 #plenty of time to converge; can be reduced for faster simulations
    samples_per_trajectory=int(restartConfigurationEvery/saveEveryConfigs)

    savedSamples = args.num_trajectories*samples_per_trajectory


    #Output directory creation
    if args.no_smcs:
        out_dir=out_dir_name(save_folder, GPU,args.replicated_length, N, args.col_rate, base_stochasticity, steps_per_sample, args.trunc, args.pull_force, args.top_monomer, args.segregated)      
    else:
        out_dir=out_dir_name_smcs(save_folder,GPU,args.replicated_length, args.fall_rate_fork, N, args.col_rate,separation, parSstrengths[0], life_time, base_stochasticity, steps_per_sample, args.trunc, args.pull_force)      
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    

    #Actually start the simulations
    if args.no_smcs:
        print("Start sims no smcs")
        sim.run_simulation(args.replicated_length,monomer_size,monomer_wig, N,\
            steps_per_sample, radius,\
            smcBondDist, smcBondWiggleDist, out_dir, saveEveryConfigs, \
            savedSamples, restartConfigurationEvery,GPU_choice=GPU,\
            F_z=args.pull_force,col_rate=args.col_rate, trunc=args.trunc, num_tethers=args.num_tethers,\
            top_monomer=args.top_monomer, start_segregated=args.segregated, forks_bound=args.forks_bound)

    else:
        print("Start sims with smcs")
        sim.run_simulation(args.replicated_length,monomer_size,monomer_wig,knockOffProb, kOriToTer_kTerToOri_kBypass, parSsites,\
                wind, terSites, terSiteStrength, parSstrengths,\
                separation,life_time, N, N_1D,base_stochasticity, translocator_initialization_steps, \
                steps_per_sample, radius, smcStepsPerBlock, \
                smcBondDist, smcBondWiggleDist, out_dir, saveEveryConfigs, \
                savedSamples, restartConfigurationEvery,\
                GPU_choice=GPU, F_z=args.pull_force, col_rate=args.col_rate,\
                trunc=args.trunc, stallFork=args.fall_rate_fork, num_tethers=args.num_tethers,\
                top_monomer=args.top_monomer, start_segregated=args.segregated)


main()
