import numpy as np
import pyximport; pyximport.install(
    setup_args={"include_dirs":np.get_include()},
    reload_support=True)
from looplib_bacterial import replicating_bacterial_no_bypassing
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

def save_to_h5(l_sites, r_sites, forks, ts, filename='data.h5'):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('l_sites', data=l_sites)
        f.create_dataset('r_sites', data=r_sites)
        f.create_dataset('forks', data=forks)
        f.create_dataset('ts', data=ts)

def out_dir_name_1D(bacterium, M, GPU, results_dir="Results_1D"):
    return os.path.join(results_dir, f"Replicating_no_bypass_{bacterium.name}_N_{bacterium.N}_M_{M}_loopsize_{bacterium.loopSize:.4g}_ter_size_{bacterium.terLength}_ter_strength_{bacterium.terStrength}_GPU_{GPU}")

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

def run_replicating_1D_sims(bacterium, num_smcs, burn_in_time_mins, sim_time_mins, num_sims, delta_t_sec=2, GPU=0, skip_gens=0):
    print(f"Performing {num_sims} replicating simulations, with dt={delta_t_sec}s")
    delta_t=delta_t_sec/60 #all rates are in minutes

    assert sim_time_mins%delta_t_sec==0, "Simulation time must be a multiple of delta_t"

    results_dir=out_dir_name_1D(bacterium, num_smcs, GPU)
    check_or_create_directory(results_dir)

    for i in range(num_sims+skip_gens):
        p = {}
        p['L'] = bacterium.N
        p['N'] = num_smcs
        p['R_OFF'] =  np.concatenate((bacterium.offloadingRates, bacterium.offloadingRates))
        p['R_ON'] = np.concatenate((bacterium.loadingRates, bacterium.loadingRates)) #no loading on unreplicated implemented in sampling
        p['R_EXTEND'] = float(bacterium.stepRate)
        p['R_SHRINK'] = float(bacterium.backstepRate)#float(.4)
        p['R_FORK'] = float(bacterium.rateReplication)
        p['T_MAX'] = sim_time_mins
        p['N_SNAPSHOTS'] = p['T_MAX']//delta_t
        p['PROCESS_NAME'] = b'replicating'
        if i>0:
            #not the first generation; inherit the sites from the previous generation
            p['INIT_L_SITES']=prev_l_sites
            p['INIT_R_SITES']=prev_r_sites
            p['BURNIN_TIME'] = 1 #no burn-in time basically; previous positions are loaded
        else:
            #first generation; add loop-extruders one by one to speed up copnvergence
            p['ACTIVATION_TIMES']=np.concatenate((np.linspace(0,burn_in_time_mins/4, num=num_smcs), np.zeros(num_smcs)))
            p['BURNIN_TIME'] = burn_in_time_mins

        t_start=time.perf_counter()
        l_sites, r_sites, forks, ts = replicating_bacterial_no_bypassing.simulate(p, verbose=False) #perform a new simulation
        t_end = time.perf_counter()

        prev_l_sites, prev_r_sites = pick_initial_sites(l_sites[-1,:], r_sites[-1,:], num_smcs, bacterium.N)

        if i>=skip_gens:
            save_to_h5(l_sites, r_sites, forks, ts, os.path.join(results_dir, f"simulation_{i-skip_gens}.h5"))

    return results_dir
