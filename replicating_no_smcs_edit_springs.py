from src.simulations_3D.replicating_no_smcs_edit_springs import run_simulation
from src.bacterial_species import *
import os, argparse


def create_parser():
    parser = argparse.ArgumentParser(description='replicating chromosome simulations, no smcs. Change spring lenghts around forks.')
    parser.add_argument('GPU', type=int, nargs='?', default=0, help="Number for GPU to use (not needed with --cpu)")
    parser.add_argument('--cpu', action='store_true', help="Run on CPU instead of CUDA (for testing without GPU)")
    parser.add_argument('--num_tethers', default=2, type=int, help="Number of tethers; 0,1,2")
    parser.add_argument('-N', '--num_monomers', default=1620, type=int, help="Number of monomers in the polymer")
    parser.add_argument('-b', '--monomer_size', default=50,type=float, help="Monomer size in nm")
    parser.add_argument('--monomer_wiggle',default=30,type=float, help="Monomer wiggle length in nm")
    parser.add_argument('-m', '--mass', default=25,type=float, help="Mass of each particle")
    parser.add_argument('--steps_per_sample', default=3500,type=int, help="How many 3D steps per SMC update.")
    parser.add_argument('-c', '--col_rate', default=0.0075,type=float, help="Collision rate that sets drag force")
    parser.add_argument('--d_l_behind', default=240,type=float, help="length across which length changes behind forks decay")
    parser.add_argument('--d_l_ahead', default=40,type=float, help="length across which length changes ahead of forks decay")
    parser.add_argument('--f_ahead', default=0.9,type=float, help="factor of length changes ahead of forks")
    parser.add_argument('--f_behind', default=1.1,type=float, help="factor of length changes behind forks")
    parser.add_argument('--top_monomer',type=int, default=0, help="The monomer that will be at the pole for an initial configuration")
    parser.add_argument('-t', '--trunc', default=5.0,type=float, help="trunc parameter for strength of excluded volume")
    parser.add_argument('-f', '--force', default=10.0,type=float, help="The force for the confinement")
    parser.add_argument('-n', '--num_trajectories', default=100,type=int, help="The number of trajectories to simulate")
    parser.add_argument('--compression_t', default=5,type=float, help="time at which compression ahead of forks starts, in minutes")
    return parser

def out_folder_name(bacterium, args_in, parent_folder="Results_3D/Replicating/"):
    return os.path.join(parent_folder, f"GPU_{args_in.GPU}_No_smcs_sph_cap_N_{args_in.num_monomers}_L_{bacterium.L_0}_sps_{args_in.steps_per_sample}_colrate_{args_in.col_rate}_trunc_{args_in.trunc}_num_tethers_{args_in.num_tethers}_dlb_{args_in.d_l_behind}_fb_{args_in.f_behind}_dla_{args_in.d_l_ahead}_fa_{args_in.f_ahead}_c_time_{args_in.compression_t}_b_{args_in.monomer_size}_wig_{args_in.monomer_wiggle}_mass_{args_in.mass}")


def main():
    parser=create_parser()
    args = parser.parse_args()

    #Basic set up; bacterium and important parameters
    GPU=args.GPU
    platform = "CPU" if args.cpu else "cuda"
    N=args.num_monomers
    monomer_size=args.monomer_size #nm; in this case for 1 kb
    #bacterium=e_coli(N,monomer_size, 1)
    bacterium=c_crescentus(N,monomer_size,1)

    #1D simulation parameters. For replicating simulations, simulation time always 1.2 * replication time
    total_time_min=int(bacterium.inferred_time(1.05*N))
    delta_t_sec=1 # for how often LE positions are saved
    smcSteps=int(total_time_min*60/delta_t_sec)

    #3D simulation parameters
    save_folder=out_folder_name(bacterium, args)
    os.makedirs(save_folder, exist_ok=True)

    delta_t_3D=60 #delta_t_3D/delta_t_sec is how many seconds there are between 3D samples
    rel_bond_wiggle=args.monomer_wiggle/monomer_size #how much the bonds can flex
    smcBondDist=50/monomer_size #condensins are roughly 50 nm in size
    saveEveryConfigs=delta_t_3D//delta_t_sec

    #RUN SIMULATIONS

    run_simulation(bacterium, rel_bond_wiggle, smcBondDist, 0.5*smcBondDist,\
        args.steps_per_sample, save_folder, smcSteps, args.num_trajectories,\
        delta_t_sec, saveEveryConfigs, GPU_choice = GPU, F_z=args.force,\
        col_rate=args.col_rate, trunc=args.trunc, top_monomer=args.top_monomer, \
        num_tethers=args.num_tethers, decay_l_behind=args.d_l_behind, factor_behind=args.f_behind,\
        decay_l_ahead=args.d_l_ahead, factor_ahead=args.f_ahead, compression_t=args.compression_t,mass=args.mass,
        platform=platform)

main()

