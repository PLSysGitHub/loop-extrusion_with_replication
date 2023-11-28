import argparse
import sys
import os
from itertools import product

def create_parser():
    parser = argparse.ArgumentParser(description='Simulations of a linear chain without confinement. Used for R^2(s) scalings.')
    parser.add_argument('GPU', type=int, nargs=1, help="Number for GPU to use")
    parser.add_argument('-c', '--col_rate', default=0.03,type=float, help="Collision rate that sets drag force")
    parser.add_argument('-t', '--trunc', default=1.5,type=float, help="trunc parameter for strength of excluded volume")
    parser.add_argument('-n', '--num_trajectories', default=2,type=int, help="The number of trajectories to simulate")
    return parser
 

def out_dir_name(save_folder, GPU_num,N, col_rate, stoch,steps_per_sample, trunc):
    fname = f"GPU_{GPU_num}_linear_N_{N}_colRate_{col_rate}"+ \
            f"_stoch_{stoch}_sps_{steps_per_sample}_trunc_{trunc}"
    return os.path.join(save_folder,fname)

def main():
    parser=create_parser()
    args = parser.parse_args()
    GPU=args.GPU[0]
    import loop_extrusion_replication.steady_state_simulations.no_smcs_no_confinement as sim
    save_folder = "Linear_No_confinement_No_smcs_no_tether"
    print("Running simulations with linear polymer without confinement")
            
    #Set rest of parameters
    
    #Dimensions
    N=4040 #number of 3D monomers. Intentionally larger than used for simulations, so we get larger range for scaling.

    monomer_size=129#nm, Messelink et.al. 2021
    monomer_wig=17**0.5/monomer_size #var for distance between 10 kb is 17 nm
    
    #1D simulation rates
    base_stochasticity= 0.05 #loop extruder step rate per simulation step

    # saving for polymer simulation
    steps_per_sample = 2500 #polymer simulation steps for SMC update
    saveEveryConfigs = 500

    restartConfigurationEvery =500000 #can be reduced for faster simulations
    samples_per_trajectory=int(restartConfigurationEvery/saveEveryConfigs)

    savedSamples = args.num_trajectories*samples_per_trajectory

    #Output directory creation
    out_dir=out_dir_name(save_folder, GPU, N, args.col_rate, base_stochasticity, steps_per_sample, args.trunc)    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    

    #Actually start the simulations
    print("Start sims free linear polymer")
    sim.run_simulation(monomer_size,monomer_wig, N,\
        steps_per_sample, out_dir, saveEveryConfigs, \
        savedSamples, restartConfigurationEvery,GPU_choice=GPU,\
        col_rate=args.col_rate, trunc=args.trunc)

main()
