from src.simulations_3D.steady_state_no_smcs import run_simulation
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
    parser.add_argument('--radius_factor', default=1.0,type=float, help="Change confinement radius by factor")
    parser.add_argument('--length_factor', default=1.0,type=float, help="Change confinement length by factor")
    parser.add_argument('--top_monomer',type=int, default=0, help="The monomer that will be at the pole for an initial configuration")
    parser.add_argument('-t', '--trunc', default=5.0,type=float, help="trunc parameter for strength of excluded volume")
    parser.add_argument('-f', '--force', default=10.0,type=float, help="The force for the confinement")
    parser.add_argument('-n', '--num_trajectories', default=5,type=int, help="The number of trajectories to simulate")
    parser.add_argument('--no_confinement',action="store_true", help="no cylindrical confinement") 
    parser.add_argument('--infinite_tube',action="store_true", help="no length constraint for confinement") 
    parser.add_argument('--add_tether',action="store_true", help="tether the top monomer to a pole") 
    return parser

def out_folder_name(bacterium, args_in):
    if args_in.no_confinement:
        parent_folder="Results_3D/Steady_state_no_confinement/"
    elif args_in.add_tether:
        parent_folder=f"Results_3D/Steady_state_tether_monomer_{args_in.top_monomer}/"
    elif args_in.infinite_tube:
        parent_folder="Results_3D/Steady_state_infinite_tube/"
    else:
        parent_folder="Results_3D/Steady_state/"

    return os.path.join(parent_folder, f"GPU_{args_in.GPU}_No_smcs_N_{args_in.num_monomers}_r_{bacterium.radius}_L_{bacterium.L_0}_num_smcs_0_loop_size_0_colrate_{args_in.col_rate}_trunc_{args_in.trunc}_ter_size_800_ter_strength_100.0_mass_{args_in.mass}")


def main():
    parser=create_parser()
    args = parser.parse_args()

    #Basic set up; bacterium and important parameters
    GPU=args.GPU
    platform = "CPU" if args.cpu else "cuda"
    N=args.num_monomers
    monomer_size=args.monomer_size #nm; in this case for 1 kb
    bacterium=c_crescentus(N,monomer_size,0)

    #If we want to perform simulations where we change dimensions
    if args.radius_factor!=1.0:
        bacterium.multiply_radius(args.radius_factor)

    if args.length_factor!=1.0:
        bacterium.multiply_length(args.length_factor)

    num_sims=args.num_trajectories

    #1D simulation parameters
    simulation_time_min=int(bacterium.inferred_time(N)*20) #inferred_time(N)=replication of chromosome, essentially a generation
    delta_t_sec=1 # for how often LE positions are saved
    num_steps_1D=simulation_time_min*60//delta_t_sec

    #3D simulation parameters
    save_folder=out_folder_name(bacterium, args)
    os.makedirs(save_folder, exist_ok=True)

    delta_t_3D=60 #delta_t_3D/delta_t_sec is how many seconds there are between 3D samples
    rel_bond_wiggle=args.monomer_wiggle/monomer_size #how much the bonds can flex

    run_simulation(bacterium, rel_bond_wiggle, args.steps_per_sample, save_folder, num_steps_1D, num_sims, delta_t_3D, GPU_choice=GPU, F_z=args.force, col_rate=args.col_rate, trunc=args.trunc, top_monomer=args.top_monomer, infinite_tube=args.infinite_tube, no_confinement=args.no_confinement, add_tether=args.add_tether, mass=args.mass,
        platform=platform)

main()

