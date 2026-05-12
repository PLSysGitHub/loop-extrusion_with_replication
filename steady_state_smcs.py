from src.simulations_3D.steady_state_smcs import run_simulation
from src.bacterial_species import *
import os, argparse


def create_parser():
    parser = argparse.ArgumentParser(description='Steady state simulations of E coli chromosome, unreplicating')
    parser.add_argument('GPU', type=int, nargs='?', default=0, help="Number for GPU to use (not needed with --cpu)")
    parser.add_argument('--cpu', action='store_true', help="Run on CPU instead of CUDA (for testing without GPU)")
    parser.add_argument('-N', '--num_monomers', default=1620, type=int, help="Number of monomers in the polymer")
    parser.add_argument('-b', '--monomer_size', default=50,type=float, help="Monomer size in nm")
    parser.add_argument('--monomer_wiggle',default=30,type=float, help="Monomer wiggle length in nm")
    parser.add_argument('-m', '--mass', default=25,type=float, help="Mass of each particle")
    parser.add_argument('--steps_per_sample', default=3500,type=int, help="How many 3D steps per SMC update.")
    parser.add_argument('-c', '--col_rate', default=0.0075,type=float, help="Collision rate that sets drag force")
    parser.add_argument('-l', '--loop_size', default=1200,type=float, help="Sets the lifetime, loop-size in kb.")
    parser.add_argument('--radius_factor', default=1.0,type=float, help="Change confinement radius by factor")
    parser.add_argument('--length_factor', default=1.0,type=float, help="Change confinement length by factor")
    parser.add_argument('--ter_size', default=800,type=int, help="Length of ter region in kb. Default 800, if lower, make chromosome shorter!")
    parser.add_argument('--ter_strength', default=100,type=float, help="The relative strength of offloading at ter region. Default 100")
    parser.add_argument('-M', '--num_smcs',type=int, default=40, help="Number of loop-extruders on single chromosome")
    parser.add_argument('--top_monomer',type=int, default=0, help="The monomer that will be at the pole for an initial configuration")
    parser.add_argument('-t', '--trunc', default=5.0,type=float, help="trunc parameter for strength of excluded volume")
    parser.add_argument('-f', '--force', default=10.0,type=float, help="The force for the confinement")
    parser.add_argument('-n', '--num_trajectories', default=5,type=int, help="The number of trajectories to simulate")
    parser.add_argument('--no_bypass',action='store_true', help="don't allow LE bypassing") 
    parser.add_argument('--no_confinement',action="store_true", help="no cylindrical confinement") 
    parser.add_argument('--infinite_tube',action="store_true", help="no length constraint for confinement") 
    parser.add_argument('--add_tether',action="store_true", help="tether the top monomer to a pole") 
    return parser

def out_folder_name(bacterium, args_in):
    if args_in.no_confinement:
        parent_folder="Results_3D/Steady_state_no_confinement/"
    elif args_in.top_monomer!=1150:
        parent_folder=f"Results_3D/Steady_state_start_top_monomer_{args_in.top_monomer}/"
    elif args_in.add_tether:
        parent_folder=f"Results_3D/Steady_state_tether_monomer_{args_in.top_monomer}/"
    elif args_in.infinite_tube:
        parent_folder="Results_3D/Steady_state_infinite_tube/"
    elif args_in.no_bypass:
        parent_folder="Results_3D/Steady_state_no_bypass/"
    else:
        parent_folder="Results_3D/Steady_state/"

    return os.path.join(parent_folder, f"GPU_{args_in.GPU}_N_{args_in.num_monomers}_{bacterium.name}_r_{bacterium.radius}_L_{bacterium.L_0}_num_smcs_{args_in.num_smcs}_loop_size_{args_in.loop_size}_colrate_{args_in.col_rate}_trunc_{args_in.trunc}_ter_size_{args_in.ter_size}_ter_strength_{args_in.ter_strength}")


def main():
    parser=create_parser()
    args = parser.parse_args()
    if args.no_bypass:
        from src.simulations_1D.one_dimension_bacterial_no_bypass import run_1D_sims
    else:
        from src.simulations_1D.one_dimension_bacterial_bypass import run_1D_sims

    #Basic set up; bacterium and important parameters
    GPU=args.GPU
    platform = "CPU" if args.cpu else "cuda"
    N=args.num_monomers
    loop_size=args.loop_size
    monomer_size=args.monomer_size #nm; in this case for 1 kb
    bacterium=c_crescentus(N,monomer_size,loop_size)

    #If we want to perform simulations where we change dimensions
    if args.radius_factor!=1.0:
        bacterium.multiply_radius(args.radius_factor)

    if args.length_factor!=1.0:
        bacterium.multiply_length(args.length_factor)

    num_smcs=args.num_smcs
    num_sims=args.num_trajectories

    #1D simulation parameters
    simulation_time_min=bacterium.inferred_time(N)*20 #inferred_time(N)=replication of chromosome, essentially a generation
    burn_in_time_min=bacterium.inferred_time(N*5)
    delta_t_sec=1 # for how often LE positions are saved

    #3D simulation parameters
    save_folder=out_folder_name(bacterium, args)
    os.makedirs(save_folder, exist_ok=True)

    delta_t_3D=60 #delta_t_3D/delta_t_sec is how many seconds there are between 3D samples
    smcBondDist=50/monomer_size #condensins are roughly 50 nm in size
    rel_bond_wiggle=args.monomer_wiggle/monomer_size #how much the bonds can flex

    #RUN SIMULATIONS
    results_dir_1D=run_1D_sims(bacterium, num_smcs, burn_in_time_min,\
        simulation_time_min, num_sims, delta_t_sec, GPU)

    run_simulation(results_dir_1D, bacterium, rel_bond_wiggle, args.steps_per_sample,\
        smcBondDist, 0.5*smcBondDist, save_folder,delta_t_3D, GPU_choice=GPU, \
        F_z=args.force, col_rate=args.col_rate, trunc=args.trunc,  mass=args.mass,\
        top_monomer=args.top_monomer, infinite_tube=args.infinite_tube, \
        no_confinement=args.no_confinement, add_tether=args.add_tether,\
        dt_1D=delta_t_sec/60, platform=platform)

main()

