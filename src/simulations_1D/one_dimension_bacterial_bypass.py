import numpy as np
import pyximport; pyximport.install(
    setup_args={"include_dirs":np.get_include()},
    reload_support=True)
from looplib_bacterial import bacterial_bypassing
import os, sys, time, shutil
import h5py
from ..bacterial_species import cell

def check_or_create_directory(directory_path):
    if os.path.exists(directory_path):
        user_input = input(f"The directory '{directory_path}' already exists. Overwrite? (y/n): ").strip().lower()
        if user_input == 'y' or user_input=="yes":
            print(f"Overwriting the directory: {directory_path}")
            shutil.rmtree(directory_path)
            os.makedirs(directory_path)
        else:
            print("Operation aborted. Exiting.")
            sys.exit()
    else:
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")

def save_to_h5(l_sites, r_sites, ts, filename='data.h5'):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('l_sites', data=l_sites)
        f.create_dataset('r_sites', data=r_sites)
        f.create_dataset('ts', data=ts)

def pick_initial_sites(l_sites, r_sites, M, L):
    assert len(l_sites)==len(r_sites), "Different number of left and right sites"
    assert len(l_sites)>M, "Number of loop-extruders from previous generation must be larger than M"
    assert np.max(l_sites)<2*L and np.max(r_sites)<2*L, "More lattice sites than expected"
    assert np.min(l_sites)>=-1 and np.min(r_sites)>=-1, "Negative lattice sites lower than -1"

    #We need to pick M loop-extruders that have legs on the same chromosome (sites<L or sites>=L), out of 2M.
    #First, exclude unbound loop-extruders, with positions -1
    bound_l_sites = l_sites[l_sites>-1]
    bound_r_sites = r_sites[r_sites>-1]

    #Then, count number of loop-extruders on each chromosome
    num_l_chrom1 = np.sum(bound_l_sites<L)
    num_l_chrom2 = np.sum(bound_l_sites>=L)

    if num_l_chrom1>=num_l_chrom2:
        picked_smcs = (bound_l_sites<L)
    else:
        picked_smcs = (bound_l_sites>=L)

    #project onto chromosome 0
    picked_l=bound_l_sites[picked_smcs]%L
    picked_r=bound_r_sites[picked_smcs]%L

    if len(picked_l)<M:
        #not enough loop-extruders on one chromosome; the rest will start unbound
        picked_l = np.concatenate((picked_l, -np.ones(M-len(picked_l))))
        picked_r = np.concatenate((picked_r, -np.ones(M-len(picked_r))))
    elif len(picked_l)>M:
        #too many loop-extruders on one chromosome; pick M at random
        indices = np.random.choice(len(picked_l), M, replace=False)
        picked_l = picked_l[indices]
        picked_r = picked_r[indices]

    return picked_l, picked_r

def out_dir_name_1D(bacterium, M, GPU, results_dir="Results_1D"):
    return os.path.join(results_dir, f"GPU_{GPU}_Unreplicating_{bacterium.name}_N_{bacterium.N}_M_{M}_loopsize_{bacterium.loopSize}_ter_size_{bacterium.terLength}_ter_strength_{bacterium.terStrength}")

def run_1D_sims(bacterium, num_smcs, burn_in_time_min, simulation_time_min, num_sims, delta_t_sec=1, GPU=0):
    delta_t=delta_t_sec/60 #units are all in minutes

    bypass_rate=float(1./(bacterium.timeTraverse+1/bacterium.stepRate)) #time is stall time plus mean step time

    results_dir=out_dir_name_1D(bacterium, num_smcs, GPU)
    check_or_create_directory(results_dir)

    for i in range(num_sims):
        p = {}
        p['L'] = bacterium.N
        p['N'] = num_smcs
        p['R_OFF'] =  bacterium.offloadingRates
        p['R_ON'] = bacterium.loadingRates
        p['R_EXTEND'] = float(bacterium.stepRate)
        p['R_SHRINK'] = float(bacterium.backstepRate)#float(.4)
        p['R_BYPASS'] = bypass_rate

        p['T_MAX'] = simulation_time_min 
        p['BURNIN_TIME'] = burn_in_time_min
        p['N_SNAPSHOTS'] = p['T_MAX']//delta_t
        p['PROCESS_NAME'] = b'proc'

        t_start=time.perf_counter()
        l_sites, r_sites, ts = bacterial_bypassing.simulate(p, verbose=False) #perform a new simulation
        t_end = time.perf_counter()

        save_to_h5(l_sites, r_sites, ts, os.path.join(results_dir, f"simulation_{i}.h5"))

    return results_dir
