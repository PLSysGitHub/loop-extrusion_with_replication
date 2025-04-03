import numpy as np
from polychrom.starting_conformations import grow_cubic

class cell:
    """
    Container class that stores information about a bacterial system, including the confinement (nucleoid) size and loop-extrusion parameters.


    Length units are in simulation monomer size,
    (approximate) time units are in minutes
    """

    def __init__(self, name, N, monomerSize, radius, L_0, growthRate, LE_lifetime, LE_stepRate, LE_backstepRate, parS_degrees, parS_strength, terRegion, terStrength, tReplication, timeTraverse=7/60):
        
        assert len(parS_degrees)==len(parS_strength)
        self.name=name
        self.N=N
        self.terStrength=terStrength
        self.terLength=len(terRegion)
        #lengths in lattice units
        self.monomer_size=monomerSize
        self.radius=radius/monomerSize
        self.L_0=L_0/monomerSize

        #rates and times using minutes
        self.rateGrowth=growthRate
        self.rateReplication=N/tReplication/2 #step rate for forks, monomers per minute

        #Calculate arrays for LE simulations
        self.loopSize=LE_lifetime*2*LE_stepRate #lifetime in minutes, LE steprate in monomers/minute
        self.offloadingRates=np.ones(N)*1/LE_lifetime #1/minute
        self.offloadingRates[terRegion]*=terStrength
    
        parSsites=[int((parS%360)*N/360) for parS in parS_degrees]
        self.loadingRates=np.ones(N)
        self.loadingRates[parSsites]=parS_strength
        if terStrength>1:
            #if we have enhanced off-loading at ter, also don't load SMCs there
            self.loadingRates[terRegion]=0

        self.stepRate=LE_stepRate
        self.backstepRate=LE_backstepRate
        self.timeTraverse=timeTraverse #minutes

    def multiply_length(self, length_factor):
        self.L_0*=length_factor

    def multiply_radius(self, radius_factor):
        self.radius*=radius_factor

    #for replicating sims. 
    def start_point_replicating(self,topMonomer):
        height=self.R_to_height(0)
        start_data = grow_cubic(self.N, (int(height)-1), method="extended")  # creates a compact conformation that fills the height
    
        for i in range(self.N):
            start_data[i][:]=(start_data[i][:]-int(height-1)/2) #center it

        #roll the array so that top monomer gets the position of monomer 1
        if topMonomer>0:
            start_data=np.roll(start_data, self.N-topMonomer)


        #add an array for the unreplicated parts
        start_data=np.vstack((start_data,start_data+np.sqrt(1/6))) #arbitrary; monomer positions initialized during replication
        
        return start_data

    #for steady state, only circular chromosome and then linear replicated segment
    def start_point_unreplicating(self,topMonomer):
        height=self.R_to_height(0)
        start_data = grow_cubic(self.N, (int(height)-1), method="extended")  # creates a compact conformation that fills the height
    
        for i in range(self.N):
            start_data[i][:]=(start_data[i][:]-int(height-1)/2) #center it

        #roll the array so that top monomer gets the position of monomer 1
        if topMonomer>0:
            start_data=np.roll(start_data, self.N-topMonomer)
        
        return start_data

    def t_to_height(self, t):
        height=self.L_0*np.exp(self.rateGrowth*t) #in lattice units
        return height

    def t_to_z_oris(self,t):
        #this function approximates ori positions in C. crescentus

        v_0=328/self.monomer_size #monomer lengths per min
        v_f=19.4/self.monomer_size #units per min
        deceleration=-30.9/self.monomer_size #units per min^2
        
        height=self.t_to_height(t)

        if t<10:#decelerating separation
            ori_sep= v_0*t +1/2*deceleration*t**2
        else:#constant speed separation
            ori_sep=v_0*10+1/2*deceleration*10**2+v_f*(t-10)

        z1=-height/2+343/self.monomer_size
        old_ori=list([0.,0.,z1])

        z2=min(height/2-343/self.monomer_size, z1+ori_sep)
        new_ori=list([0.,0.,z2])

        return old_ori, new_ori


#E coli
def e_coli(N, monomer_size, loop_size, ter_size_kb=800, terStrength=100, no_bypass=False):
    t_replication=70 #minutes
    t_doubling=100
    growth_rate=np.log(2)/t_doubling #exponential growth rate for cells, per minute
    loop_extruder_speed = 18#46 #kb/min
    radius=1000/2 #nm
    
    bp_per_monomer=N//4600
    cut_from_ter=800-ter_size_kb
    total_kb=4600-cut_from_ter
    N_cut=N-cut_from_ter//bp_per_monomer

    half_ter_region=ter_size_kb//2//bp_per_monomer
    mid_ter_region=N_cut//2
    
    ter_sites = np.arange(mid_ter_region-half_ter_region,mid_ter_region+half_ter_region) #ter region centered around ter
    parS=[] #no parS sites
    parS_strength=[]#no parS sites
    L_0=1600*np.exp(growth_rate*(t_doubling-t_replication))#cell size at start of replication, nm. cells grow 10 minutes before replication starts
    lifetime=loop_size/(2*loop_extruder_speed)
    stepRate=loop_extruder_speed/(4600/N)
    backStepRate=0
    if no_bypass:
        return cell("e_coli_no_bypass",N_cut, monomer_size, radius,L_0,growth_rate,lifetime,stepRate,backStepRate,parS,parS_strength,ter_sites,terStrength,t_replication, 0)
    else:
        return cell("e_coli",N_cut, monomer_size, radius,L_0,growth_rate,lifetime,stepRate,backStepRate,parS,parS_strength,ter_sites,terStrength,t_replication)

#Caulobacter
def c_crescentus(N,monomer_size,loop_size, terStrength=100, frac_load_ori=0.5, time_trav_min=7/60):
    total_kb=4050
    t_replication=85 #minutes. to match 4D-MaxEnt model.75 minutes according to Skerker & Laub 2004
    loop_extruder_speed = 18 #kb/min. Tran et al. 2017
    t_doubling=125 #Skerker&Laub, 2004
    growth_rate=np.log(2)/t_doubling #exponential growth rate for cells, per minute
    radius=750/2 #nm. Messelink et al. 2021
    ter_sites = np.arange(N*195//405,N*205//405) #ter region bounds on the 1D lattice
    parS_strength=[N*frac_load_ori/(1-frac_load_ori)] #relative affinity of parS sites to rest of chromosome, when other sites have strength 1
    parS=[0] #degrees
    L_0=2252#cell size at t=0, nm. Messelink et. al. 2021
    lifetime= loop_size/(2*loop_extruder_speed) #in minutes
    stepRate=loop_extruder_speed/(total_kb/N) #monomers per minute
    backStepRate=0
    return cell("caulobacter", N,monomer_size,radius,L_0,growth_rate,lifetime,stepRate,backStepRate,parS,parS_strength,ter_sites,terStrength,t_replication, time_trav_min)


