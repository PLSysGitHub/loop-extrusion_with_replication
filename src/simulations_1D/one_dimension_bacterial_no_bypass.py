import numpy as np
import pyximport; pyximport.install(
    setup_args={"include_dirs":np.get_include()},
    reload_support=True)
from looplib_bacterial import bacterial_no_bypassing
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

def out_dir_name_1D(bacterium, M, results_dir="Results_1D"):
    return os.path.join(results_dir, f"Unreplicating_{bacterium.name}_N_{bacterium.N}_M_{M}_loopsize_{bacterium.loopSize:.4g}_ter_size_{bacterium.terLength}_ter_strength_{bacterium.terStrength}")

def run_1D_sims(bacterium, num_smcs, ter_size, burn_in_time_min, simulation_time_min, num_sims, delta_t_sec=1):
    delta_t=delta_t_sec/60 #all rates are in minutes

    results_dir=out_dir_name_1D(bacterium, num_smcs)
    check_or_create_directory(results_dir)

    for i in range(num_sims):
        p = {}
        p['L'] = bacterium.N
        p['N'] = num_smcs
        p['R_OFF'] =  bacterium.offloadingRates
        p['R_ON'] = bacterium.loadingRates
        p['R_EXTEND'] = float(bacterium.stepRate)
        p['R_SHRINK'] = float(bacterium.backstepRate)#float(.4)

        p['T_MAX'] = simulation_time_min 
        p['BURNIN_TIME'] = burn_in_time_min
        p['N_SNAPSHOTS'] = p['T_MAX']//delta_t
        p['PROCESS_NAME'] = b'proc'

        t_start=time.perf_counter()
        l_sites, r_sites, ts = bacterial_no_bypassing.simulate(p, verbose=False) #perform a new simulation
        t_end = time.perf_counter()

        save_to_h5(l_sites, r_sites, ts, os.path.join(results_dir, f"simulation_{i}.h5"))

    return results_dir
