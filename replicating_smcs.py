from src.simulations_3D.replicating_smcs import run_simulation
from src.bacterial_species import *
import os, argparse

def create_parser():
    parser = argparse.ArgumentParser(description='replicating simulations of bacterial chromosome, with SMCs')
    parser.add_argument('GPU', type=int, nargs=1, help="Number for GPU to use")
    parser.add_argument('-N', '--num_monomers', default=1620, type=int, help="Number of monomers in the polymer")
    parser.add_argument('-b', '--monomer_size', default=50,type=float, help="Monomer size in nm")
    parser.add_argument('--monomer_wiggle',default=30,type=float, help="Monomer wiggle length in nm")
    parser.add_argument('-m', '--mass', default=25,type=float, help="Mass of each particle")
    parser.add_argument('--steps_per_sample', default=3500,type=int, help="How many 3D steps per SMC update.")
    parser.add_argument('-c', '--col_rate', default=0.0075,type=float, help="Collision rate that sets drag force")
    parser.add_argument('-l', '--loop_size', default=1200,type=float, help="Sets the lifetime, loop-size in kb.")
    parser.add_argument('--ter_strength', default=100,type=float, help="The relative strength of offloading at ter region. Default 100")
    parser.add_argument('--ori_frac', default=0.5,type=float, help="Fraction loaded at ori, unreplicating")
    parser.add_argument('--t_trav', default=7,type=float, help="Time stalled during loop-extruder collisions, in seconds")
    parser.add_argument('-M', '--num_smcs',type=int, default=40, help="Number of loop-extruders on single chromosome")
    parser.add_argument('--top_monomer',type=int, default=0, help="The monomer that will be at the pole for an initial configuration")
    parser.add_argument('--num_tethers', default=2, type=int, help="Number of tethers; 0,1,2")
    parser.add_argument('-t', '--trunc', default=5.0,type=float, help="trunc parameter for strength of excluded volume")
    parser.add_argument('-f', '--force', default=10.0,type=float, help="The force for the confinement")
    parser.add_argument('-n', '--num_trajectories', default=200,type=int, help="The number of trajectories to simulate")
    parser.add_argument('--no_bypass',action="store_true", help="don't allow LE bypassing")
    parser.add_argument('--fork_bypass',action="store_true", help="loop-extruders bypass replication forks")
    parser.add_argument('-g', '--skip_gens_1D', default=20, type=int, help="Number of generations to skip in 1D sims")
    return parser

def out_folder_name(bacterium, args_in, parent_folder="Results_3D/Replicating/"):
    p= os.path.join(parent_folder, f"GPU_{args_in.GPU[0]}_N_{args_in.num_monomers}_{bacterium.name}_sph_cap_L_{bacterium.L_0}_sps_{args_in.steps_per_sample}_M_{args_in.num_smcs}_loop_size_{args_in.loop_size}_colrate_{args_in.col_rate}_trunc_{args_in.trunc}_ter_strength_{args_in.ter_strength}_num_tethers_{args_in.num_tethers}_b_{args_in.monomer_size}_wig_{args_in.monomer_wiggle}_mass_{args_in.mass}_ori_frac_{args_in.ori_frac}_skip_gens_1D_{args_in.skip_gens_1D}_t_trav_{args_in.t_trav}")

    if args_in.fork_bypass:
        p=p+"_fork_bypass"
    elif args_in.no_bypass:
        p=p+"_no_bypass"

    return p


def main():
    parser=create_parser()
    args = parser.parse_args()
    if args.fork_bypass:
        from src.simulations_1D.replicating_bacterial_fork_bypass import run_replicating_1D_sims
    elif args.no_bypass:
        from src.simulations_1D.replicating_bacterial_no_bypass import run_replicating_1D_sims
    else:
        from src.simulations_1D.replicating_bacterial_bypass import run_replicating_1D_sims

    #Basic set up; bacterium and important parameters
    GPU=args.GPU[0]
    N=args.num_monomers
    loop_size=args.loop_size
    monomer_size=args.monomer_size #nm; in this case for 1 kb
    bacterium=c_crescentus(N,monomer_size,loop_size,args.ter_strength, args.ori_frac, args.t_trav/60)
    num_smcs=args.num_smcs
    num_sims=args.num_trajectories

    #1D simulation parameters.
    burn_in_time_min=bacterium.inferred_time(N)*2
    total_time_min=80
    delta_t_sec=1 # for how often LE positions are saved

    #3D simulation parameters
    save_folder=out_folder_name(bacterium, args)
    os.makedirs(save_folder, exist_ok=True)

    delta_t_3D=60 #delta_t_3D/delta_t_sec is how many seconds there are between 3D samples
    saveEveryConfigs=delta_t_3D//delta_t_sec
    smcBondDist=50/monomer_size #condensins are roughly 50 nm in size
    rel_bond_wiggle=args.monomer_wiggle/monomer_size #how much the bonds can flex

    #RUN SIMULATIONS
    results_dir_1D=run_replicating_1D_sims(bacterium, num_smcs, burn_in_time_min, total_time_min, num_sims, delta_t_sec, GPU, args.skip_gens_1D)

    run_simulation(results_dir_1D, bacterium, rel_bond_wiggle,\
                    args.steps_per_sample, smcBondDist, 0.5*smcBondDist, save_folder,\
                    saveEveryConfigs, GPU_choice = GPU, F_z=args.force, mass=args.mass,\
                    col_rate=args.col_rate, trunc=args.trunc, top_monomer=args.top_monomer, \
                    num_tethers=args.num_tethers, dt_1D=delta_t_sec/60)

main()

