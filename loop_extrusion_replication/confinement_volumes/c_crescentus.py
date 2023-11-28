from polychrom.starting_conformations import grow_cubic, create_random_walk
import numpy as np

def start_point_unsegregated(top_monomer, R, N, height):
    start_data = grow_cubic(N, (int(height)-1), method="extended")  # creates a compact conformation that fills the height
    
    for i in range(N):
        start_data[i][:]=(start_data[i][:]-int(height-1)/2) #center it

    #roll the array so that top monomer gets the position of monomer 1
    if top_monomer>0:
        start_data=np.roll(start_data, N-top_monomer)
    
    #unsegregated; stack aligned positions for second chromosome
    start_data=np.vstack((start_data,(start_data+np.sqrt(1/6)))) #offset so potential energy doesnt blow up in your face
    return start_data

def start_point_segregated(R,N,height):
    start_data = grow_cubic(N, (int(height/2)-1), method="extended")  # creates a compact conformation that fills the height
    
    for i in range(N):
        start_data[i][:]=(start_data[i][:]-[int(height-1)/2, int(height-1)/2, start_data[R//2][2]]) #fork at z=0

    #stack aligned positions for second chromosome
    start_data=np.vstack((start_data,(start_data+np.sqrt(1/6)))) #offset so potential energy doesnt blow up in your face
    
    #flip replicated segment across xy plane
    for i in range(R//2):
        start_data[i+N][2]*=-1
        start_data[2*N-1-i][2]*=-1

    return start_data

def start_point_linear_free(N):
    #for sims without confinement; linear polymer
    start_data=create_random_walk(1.,N) #unconfined random walk

    return start_data

def R_to_height(R,N, monomer_size):
    rate_replication=N/75 #monomers per minute
    rate_growth=0.0055 #exponential growth rate for cells, per minute
    L_0=25.6*88/monomer_size #cell size at t=0, simulation units
    
    inferred_t=R/rate_replication #in minutes
    
    height=L_0*np.exp(rate_growth*inferred_t)
    return height

def R_to_z_oris(R,N,height,monomer_size):
    v_0=328/monomer_size #units per min
    rate_replication=N/75 #monomers per minute
    v_f=19.4/monomer_size #units per min
    deceleration=-30.9/monomer_size #units per min^2
    
    inferred_t=R/rate_replication #in minutes
    if inferred_t<10:#decelerating separation
        ori_sep= v_0*inferred_t +1/2*deceleration*inferred_t**2
    else:#constant speed separation
        ori_sep=v_0*10+1/2*deceleration*10**2+v_f*(inferred_t-10)

    old_ori=list([0.,0.,-height/2+343/monomer_size])
    new_ori=list([0.,0.,-height/2+343/monomer_size+ori_sep])
    if -height/2+343/monomer_size+ori_sep > height/2:
        print("ERROR! new ori out of bounds. Setting at  height/2-343/monomer_size")
        new_ori=list([0.,0.,height/2-343/monomer_size])
    return old_ori, new_ori
