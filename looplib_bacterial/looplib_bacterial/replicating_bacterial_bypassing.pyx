###cython: profile=True
##cython: boundscheck=False
##cython: wraparound=False
##cython: nonecheck=False
##cython: initializedcheck=False
from __future__ import division, print_function
cimport cython
import numpy as np
cimport numpy as np

from cpython cimport bool
from heapq import heappush, heappop

from libc.stdlib cimport rand, srand, RAND_MAX
srand(0)
np.random.seed()

cdef inline np.int64_t rand_int(int N_MAX):
    return np.random.randint(N_MAX)

cdef inline np.int64_t sample_load_point(np.float64_t [:] weights, np.int64_t [:] occupied):
    cdef np.int64_t [:] inds = np.arange(len(weights))
    cdef np.float64_t [:] ps =np.copy(weights)
    cdef np.int64_t i

    for i in occupied:
        if i >= 0:
            ps[i] = 0

    ps /= np.sum(ps)

    return np.random.choice(inds, p=ps)

cdef inline np.float64_t rand_exp(np.float64_t mean):
    return np.random.exponential(mean) if mean > 0 else 0

cdef inline float rand_float():
    return rand() / float(RAND_MAX)

cdef inline int64sign(np.int64_t x):
    if x > 0:
        return +1
    else:
        return 0

cdef inline int64abs(np.int64_t x):
    if x > 0:
        return x
    else:
        return -x

cdef inline int64not(np.int64_t x):
    if x == 0:
        return 1
    else:
        return 0

cdef class System:
    cdef np.int64_t L
    cdef np.int64_t N #this is the current number of LEs
    cdef np.int64_t N_max #how many LEs there will be on a fully replicated chromosome
    cdef np.float64_t time
    cdef np.float64_t bypass_rate #rate at which LEFs can move to occupied sites
    cdef np.float64_t fork_rate #movement rate for forks independently
    cdef np.int64_t L_replicated #distance between forks once replication started

    cdef np.float_t [:] vels #4 N_max; 2 directions, 2 legs
    cdef np.float_t [:] lifespans # 2 N_max
    cdef np.float_t [:] rebinding_times # 2 N_max
    cdef np.float_t [:] perms #L+1, assume identical for replicated

    cdef np.int64_t [:] fork #fork[0] moves left, fork[1] right
    cdef np.int64_t [:,:] lattice #First index; chromosome. Second index: lattice position.
    cdef np.int64_t [:] locs #N_max. contains indices between -1 and 2L-1. -1 unbound. between L and 2L-1 on replicated

    cdef np.float_t [:] binding_affinities #relative binding affinities; higher values mean more likely to bind. Length 2L.
    cdef np.float_t [:] unbinding_rates #unbinding rates for each site. 2L.

    def __cinit__(self, L, N, vels, rebinding_times, unbinding_rates, bypass_rate,
                  init_locs=None, perms=None, binding_rates=None):
        self.L = L
        self.L_replicated = 0
        self.N = N
        self.N_max = 2*N
        self.bypass_rate = bypass_rate

        #we start with unbound forks and no replication
        self.fork_rate=0
        self.fork=np.array([-1,-1], dtype=np.int64) #similar to LEs, first index moves left, second index right

        self.lattice = -1*np.ones((2,2*L), dtype=np.int64) #-1 means unoccupied. 0:L-1 are for left moving, L:2L-1 are for right moving. First index is chromosome.
        self.vels = vels #4 N_max; 2 directions, 2 legs
        
        self.rebinding_times = rebinding_times
        self.unbinding_rates = unbinding_rates

        if perms is None:
            self.perms = np.ones(L+1, dtype=np.float64)
        else:
            self.perms = perms

        if binding_rates is None:
            self.binding_affinities=np.ones(2*L, dtype=np.float64)
        else:
            self.binding_affinities=binding_rates

        #initialize array for ALL loop-extruders. Initially unbound.
        self.locs = -1 * np.ones(self.N_max*2, dtype=np.int64)
        
        # Initialize non-random loops
        cdef np.int64_t i, current_loc
        for i in range(self.N):
            if init_locs[i]<0 and init_locs[i+self.N]<0:
                continue
            elif init_locs[i]<0 or init_locs[i+self.N]<0:
                raise Exception('initial leg positions inconsistent')
            elif init_locs[i]>=self.L or init_locs[i+self.N]>=self.L:
                raise Exception('initial leg positions unreplicated')
            # left leg
            current_loc=init_locs[i]
            self.locs[i]=current_loc
            self.lattice[0, current_loc]=i
            # right leg
            current_loc=init_locs[i+self.N]
            self.locs[i+self.N_max]=current_loc
            self.lattice[0, current_loc+self.L]=i+self.N_max

    cdef np.int64_t move_fork(System self, np.int64_t fork_id):
        cdef np.int64_t prev_pos = self.fork[fork_id]
        cdef np.int64_t status=1

        #move the fork
        self.fork[fork_id]+= 2*fork_id-1 # -1 for zero, 1 for 1
        self.L_replicated+=1

        #potentially increase number of LEs
        self.N= self.N_max*(self.L+self.L_replicated)//(2*self.L)

        if self.fork[0]<self.fork[1]:
            status=0 #forks passed each other!
        elif self.fork[0]==self.fork[1]: #replication complete
            self.fork_rate=0

        return status


    cdef np.int64_t next_pos(System self, np.int64_t leg_idx, np.int64_t direction):
        cdef np.int64_t prev_pos = self.locs[leg_idx]
        cdef np.int64_t new_pos = (prev_pos%self.L + direction + self.L)%self.L #periodic for bacteria. within (0,L-1)

        if self.L_replicated>0 and prev_pos>=self.L: #on the replicated chromosome
            new_pos+=self.L

        return new_pos

    cdef np.int64_t make_step(System self, np.int64_t leg_idx, np.int64_t direction):
        """
        The variable `direction` can only take values +1 or -1.
        """

        cdef np.int64_t new_pos = self.next_pos(leg_idx, direction)
        return self.move_leg(leg_idx, new_pos)

    cdef np.int64_t move_leg(System self, np.int64_t leg_idx, np.int64_t new_pos):
        cdef np.int64_t lattice_increment, chrom_copy
        cdef np.int64_t prev_pos, prev_chrom_copy 

        if leg_idx >= self.N_max:
            lattice_increment=self.L #right moving leg
        else:
            lattice_increment=0 #left moving leg

        chrom_copy = int(new_pos>=self.L) #1 if on replicated chromosome, 0 otherwise

        if (new_pos >= 0) and (self.lattice[chrom_copy, new_pos%self.L + lattice_increment] >=0): #if moving to an occupied site
            print("move leg ", leg_idx, " to occupied site ", new_pos, ":",self.lattice[chrom_copy, new_pos%self.L + lattice_increment])
            return 0
        
        if self.fork_rate>0:
            if (new_pos==self.fork[1] or new_pos==self.fork[0]): #moving to a fork during replication
                print("move to fork")
                return 0

            if (new_pos>=self.fork[1]+self.L) and (new_pos<=self.fork[0]+self.L): #if moving to an unreplicated site
                print("move to unrepl")
                return 0
        
        if (new_pos>self.L*2-1) or (new_pos<-1): #if moving outside the system
            print("move outside")
            return 0

        prev_pos = self.locs[leg_idx] #between (0,L-1) for first strand, between (0,fork[1]) or (fork[0], L-1) for second strand
        prev_chrom_copy = int(prev_pos>=self.L)
        self.locs[leg_idx] = new_pos

        if (prev_pos>=0) and (new_pos>=0) and (prev_chrom_copy!=chrom_copy): #if moving between replicated and unreplicated
            print("Trying to move between replicated and unreplicated regions! Old pos: ", prev_pos, " new pos: ", new_pos)
            return 0

        if prev_pos >= 0:
            if self.lattice[prev_chrom_copy, prev_pos%self.L+lattice_increment] != leg_idx:
                print(leg_idx, " wasnt attached at prev site ", prev_pos, ". id there: ", self.lattice[prev_chrom_copy, prev_pos%self.L+lattice_increment])
                return 0
            self.lattice[prev_chrom_copy, prev_pos%self.L+lattice_increment] = -1

        if new_pos >= 0:
            self.lattice[chrom_copy, new_pos%self.L+lattice_increment] = leg_idx

        return 1

    cdef np.int64_t detach(System self, np.int64_t loop_idx):
        cdef np.int64_t status=1
        cdef np.int64_t pos1, pos2

        pos1 = self.locs[loop_idx]
        pos2 = self.locs[loop_idx+self.N_max]

        if pos1 < 0 or pos2 < 0:
            print("Trying to detach loop ", loop_idx, " which is already detached!")
            status=0
        else:
            self.move_leg(loop_idx, -1)
            self.move_leg(loop_idx+self.N_max, -1)

        return status

    cdef np.int64_t check_system(System self):
        okay = 1
        cdef np.int64_t i

        if self.fork[0]<self.fork[1]:
            print('forks are inconsistent: ', self.fork[0] , " is below ", self.fork[1])
            okay=0

        for i in range(self.N):
            if (self.locs[i] >= self.L+self.fork[1]) and (self.locs[i]<=self.L+self.fork[0]):
                print('leg ', i, 'is at an unreplicated locus: ', self.locs[i])
                okay = 0
            if (self.locs[i+self.N_max] >= self.L+self.fork[1]) and (self.locs[i+self.N_max]<=self.L+self.fork[0]):
                print('leg ', i+self.N_max, 'is at an unreplicated locus: ', self.locs[i])
                okay = 0
            if self.fork_rate>0 and (self.locs[i]==self.fork[1] or self.locs[i]==self.fork[0]):
                print('leg ', i, ' is at the fork')
                okay=0
            if self.fork_rate>0 and (self.locs[i+self.N_max]==self.fork[1] or self.locs[i+self.N_max]==self.fork[0]):
                print('leg ', i+self.N_max, ' is at the fork')
                okay=0
            if (self.locs[i]>= 2*self.L):
                print('leg ', i, 'is located outside of the system: ', self.locs[i])
                okay = 0
            if (self.locs[i+self.N] >= self.L*2):
                print('leg ', i+self.N_max, 'is located outside of the system: ', self.locs[i+self.N_max])
                okay = 0
            if (((self.locs[i] < 0) and (self.locs[i+self.N_max] >= 0 ))
                or ((self.locs[i] >= 0) and (self.locs[i+self.N_max] < 0 ))):
                print('the legs of the loop', i, 'are inconsistent: ', self.locs[i], self.locs[i+self.N_max])
                okay = 0

        for i in range(self.N, self.N_max):
            if self.locs[i]!=-1 or self.locs[i+self.N_max]:
                print('loop-extruder ', i, ' has been loaded too early')
                okay=0

        return okay

cdef class Event_t:
    cdef public np.float64_t time
    cdef public np.int64_t event_idx

    def __cinit__(Event_t self, np.float_t time, np.int64_t event_idx):
        self.time = time
        self.event_idx = event_idx

    def __richcmp__(Event_t self, Event_t other, int op):
        if op == 0:
            return 1 if self.time <  other.time else 0
        elif op == 1:
            return 1 if self.time <= other.time else 0
        elif op == 2:
            return 1 if self.time == other.time else 0
        elif op == 3:
            return 1 if self.time != other.time else 0
        elif op == 4:
            return 1 if self.time >  other.time else 0
        elif op == 5:
            return 1 if self.time >= other.time else 0


cdef class Event_heap:
    """Taken from the official Python website"""
    cdef public list heap
    cdef public dict entry_finder

    def __cinit__(self):
        self.heap = list()
        self.entry_finder = dict()

    cdef add_event(Event_heap self, np.int64_t event_idx, np.float64_t time=0):
        'Add a new event or update the time of an existing event.'
        if event_idx in self.entry_finder:
            self.remove_event(event_idx)
        cdef Event_t entry = Event_t(time, event_idx)
        self.entry_finder[event_idx] = entry
        heappush(self.heap, entry)

    cdef remove_event(Event_heap self, np.int64_t event_idx):
        'Mark an existing event as REMOVED.'
        cdef Event_t entry
        if event_idx in self.entry_finder:
            entry = self.entry_finder.pop(event_idx)
            entry.event_idx = -1

    cdef Event_t pop_event(Event_heap self):
        'Remove and return the closest event.'
        cdef Event_t entry
        while self.heap:
            entry = heappop(self.heap)
            if entry.event_idx != -1:
                del self.entry_finder[entry.event_idx]
                return entry
        return Event_t(0, 0.0)


cdef regenerate_event(System system, Event_heap evheap, np.int64_t event_idx):
    """
    Regenerate an event in an event heap. If the event is currently impossible (e.g. a step
    onto an occupied site) then the new event is not created, but the existing event is not
    modified.

    Possible events:
    0 to 2N-1 : a step to the left
    2N to 4N-1 : a step to the right
    4N to 5N-1 : passive unbinding
    5N to 6N-1 : rebinding to a randomly chosen site
    6N : move fork[0]
    6N+1 : move fork[1]
    """

    cdef np.int64_t leg_idx, loop_idx
    cdef np.int64_t direction, lattice_increment, chrom_copy
    cdef np.int64_t pos1, pos2
    cdef np.float_t local_vel
    cdef np.int64_t new_position
    cdef np.float_t rate_unbind

    if (event_idx < 4 * system.N_max) :
        if event_idx < 2 * system.N_max:
            leg_idx = event_idx
            direction = -1
        else:
            leg_idx = event_idx - 2 * system.N_max
            direction = 1

        loop_idx=leg_idx%system.N_max

        if loop_idx<system.N and (system.locs[leg_idx] >= 0):
            if leg_idx >= system.N_max:
                lattice_increment=system.L
            else:
                lattice_increment=0
            
            new_position=system.next_pos(leg_idx, direction) #periodic for bacteria, takes replication into account
            chrom_copy = int(new_position>=system.L)

            # Local velocity = velocity * permeability
            if new_position%system.L==system.fork[0] or new_position%system.L==system.fork[1]: #don't move to forks
                local_vel=0
            else:
                local_vel = (system.perms[system.locs[leg_idx]%system.L + (direction+1)//2] * system.vels[leg_idx + (direction+1)*system.N_max])

            if local_vel > 0:
                if system.lattice[chrom_copy, new_position%system.L+lattice_increment]<0: #bypassing but not overtaking
                    local_vel=np.min([system.bypass_rate, local_vel]) #move at slowest of two rates
                evheap.add_event(
                    event_idx,
                    system.time + rand_exp(1.0 / local_vel))
            else:
                evheap.remove_event(event_idx)

    # Passive unbinding.
    elif (event_idx >= 4 * system.N_max) and (event_idx < 5 * system.N_max):
        loop_idx = event_idx - 4 * system.N_max

        pos1 = system.locs[loop_idx]
        pos2 = system.locs[loop_idx+system.N_max]

        if loop_idx<system.N and (pos1 >= 0) and (pos2 >= 0): #if attached
            if (pos1==system.fork[0] or pos1==system.fork[1] or pos2==system.fork[0] or pos2==system.fork[1]):
                #unbind immediately at forks
                do_event(system, evheap, event_idx)
            else:
                rate_unbind = np.max([system.unbinding_rates[pos1], system.unbinding_rates[pos2]])
                evheap.add_event(
                    event_idx,
                    system.time + rand_exp(1. / rate_unbind))

    # Rebinding from the solution to a random site.
    elif (event_idx >= 5 * system.N_max) and (event_idx < 6 * system.N_max):
        loop_idx = event_idx - 5 * system.N_max

        if loop_idx<system.N and (system.locs[loop_idx] < 0) and (system.locs[loop_idx+system.N_max] < 0): #if detached
            evheap.add_event(
                event_idx,
                system.time + rand_exp(system.rebinding_times[loop_idx]))

    elif event_idx==6*system.N_max or event_idx == 6*system.N_max+1:
        if system.fork[0]>system.fork[1]: #ongoing but incomplete
            evheap.add_event(event_idx, system.time+rand_exp(1/system.fork_rate))

cdef regenerate_neighbours(System system, Event_heap evheap, np.int64_t pos):
    """
    Regenerate the motion events for the adjacent loop legs.
    Use to unblock the previous neighbors and block the new ones.
    """
    cdef np.int64_t nb_right, nb_left, chrom_copy

    nb_right=(pos+1)%system.L
    nb_left=(pos-1+system.L)%system.L
    chrom_copy = int(pos>=system.L)

    # regenerate left step for neighbour on the right
    if system.lattice[chrom_copy, nb_right] >= 0:
        regenerate_event(system, evheap, system.lattice[chrom_copy, nb_right])

    if system.lattice[chrom_copy, nb_right+system.L] >= 0:
        regenerate_event(system, evheap, system.lattice[chrom_copy, nb_right+system.L])

    # regenerate right step for neighbour on the left
    if system.lattice[chrom_copy, nb_left] >= 0:
        regenerate_event(system, evheap, system.lattice[chrom_copy, nb_left] + 2 * system.N_max)

    if system.lattice[chrom_copy, nb_left+system.L] >= 0:
        regenerate_event(system, evheap, system.lattice[chrom_copy, nb_left+system.L] + 2 * system.N_max)


cdef regenerate_neighbours_and_self(System system, Event_heap evheap, np.int64_t pos):
    """
    Regenerate the motion events for the adjacent loop legs as well as legs on the same site.
    Used while moving; upon bypassing, need to update rates on the same site.
    """
    cdef np.int64_t on_site, chrom_copy
    regenerate_neighbours(system, evheap, pos)

    chrom_copy=int(pos>=system.L)

    #regenerate steps for site itself. This is needed for bypassing.
    if (pos>0) and system.lattice[chrom_copy, pos%system.L] >= 0:
        on_site=system.lattice[chrom_copy, pos%system.L] #leg index on the site.
        regenerate_event(system, evheap, on_site)
        regenerate_event(system, evheap, on_site+2*system.N_max)
    
    if (pos>0) and system.lattice[chrom_copy, pos%system.L+system.L] >= 0:
        on_site = system.lattice[chrom_copy, pos%system.L+system.L] # leg index on site
        regenerate_event(system, evheap, on_site)
        regenerate_event(system, evheap, on_site + 2 * system.N_max)

 
cdef regenerate_all_loop_events(System system, Event_heap evheap,
                                np.int64_t loop_idx):
    """
    Regenerate all possible events for a loop. Includes the four possible motions and passive unbinding.
    """

    regenerate_event(system, evheap, loop_idx)
    regenerate_event(system, evheap, loop_idx + system.N_max)
    regenerate_event(system, evheap, loop_idx + 2 * system.N_max)
    regenerate_event(system, evheap, loop_idx + 3 * system.N_max)
    regenerate_event(system, evheap, loop_idx + 4 * system.N_max)
    regenerate_event(system, evheap, loop_idx + 5 * system.N_max)
    
cdef np.int64_t regenerate_after_fork_move(System system, Event_heap evheap, np.int64_t fork_idx):
    cdef np.int64_t leg_idx, loop_idx, fork_pos, move_direction, site_behind, site_ahead
    cdef np.int64_t status = 1

    fork_pos=system.fork[fork_idx]
    move_direction=2*fork_idx-1 # -1 for 0, 1 for 1

    #check if there are any LEs at the fork site. If yes, unbind them.
    if system.lattice[0, fork_pos]>=0:
        leg_idx=system.lattice[0, fork_pos]
        loop_idx=leg_idx%system.N_max
        do_event(system, evheap, 4*system.N_max+loop_idx) #detach the LEF

    if system.lattice[0, fork_pos+system.L]>=0:
        leg_idx=system.lattice[0, fork_pos+system.L]
        loop_idx=leg_idx%system.N_max
        do_event(system, evheap, 4*system.N_max+loop_idx) #detach the LEF

    if system.lattice[0, fork_pos]>=0 or system.lattice[0, fork_pos+system.L]>=0:
        print("Unbinding of LEFs at fork failed!")
        return 0

    if system.lattice[1, fork_pos]>=0 or system.lattice[1, fork_pos+system.L]>=0:
        print("There were LEFs on unreplicated site!")
        return 0

    site_ahead=fork_pos+move_direction
    # Remove moves in front of fork to fork
    if system.lattice[0, site_ahead]>=0:
        leg_idx=system.lattice[0, site_ahead]
        evheap.remove_event(leg_idx+2*system.N_max*(1-fork_idx))

    if system.lattice[0, site_ahead+system.L]>=0:
        leg_idx=system.lattice[0, site_ahead+system.L]
        evheap.remove_event(leg_idx+2*system.N_max*(1-fork_idx))

    #Regenerate movement rate two steps behind fork
    if system.fork[1]+system.L-system.fork[0]>2: #two steps behind is a part of lattice
        site_behind=(fork_pos-2*move_direction)%system.L

        if site_behind<0 or site_behind>=system.L:
            print("site_behind is ", site_behind)
            return 0

        if system.lattice[0, site_behind]>=0:
            leg_idx=system.lattice[0, site_behind]
            regenerate_event(system, evheap, leg_idx+2*system.N_max*fork_idx) #if fork moves left, regenerate left. If fork moves right, regenerate right.

        if system.lattice[0, site_behind+system.L]>=0:
            leg_idx=system.lattice[0, site_behind+system.L]
            regenerate_event(system, evheap, leg_idx+2*system.N_max*fork_idx) #if fork moves left, regenerate left. If fork moves right, regenerate right.

        if system.lattice[1, site_behind]>=0:
            leg_idx=system.lattice[1, site_behind]
            regenerate_event(system, evheap, leg_idx+2*system.N_max*fork_idx) #if fork moves left, regenerate left. If fork moves right, regenerate right.

        if system.lattice[1, site_behind+system.L]>=0:
            leg_idx=system.lattice[1, site_behind+system.L]
            regenerate_event(system, evheap, leg_idx+2*system.N_max*fork_idx) #if fork moves left, regenerate left. If fork moves right, regenerate right.

    #if we added a LEF, we need to generate its movement probabilities
    regenerate_all_loop_events(system, evheap, system.N-1)

    #regenerate fork move event
    regenerate_event(system, evheap, 6*system.N_max+fork_idx)

    return status

cdef start_replication(System system, Event_heap evheap, np.float64_t replication_rate):
    system.fork_rate=replication_rate
    system.fork[0]=system.L-1
    system.fork[1]=0
    regenerate_after_fork_move(system, evheap, 0)
    regenerate_after_fork_move(system, evheap, 1)


cdef np.int64_t do_event(System system, Event_heap evheap, np.int64_t event_idx) except 0:
    """
    Apply an event from a heap on the system and then regenerate it.
    If the event is currently impossible (e.g. a step onto an occupied site),
    it is not applied, however, no warning is raised.

    Also, partially checks the system for consistency. Returns 0 is the system
    is not consistent (a very bad sign), otherwise returns 1 if the event was a step,
    2 if the event was binding, 3 if the event was rebinding, and 4 if the event was a fork move.

    Possible events:
    0 to 2N-1 : a step to the left
    2N to 4N-1 : a step to the right
    4N to 5N-1 : passive unbinding
    5N to 6N-1 : rebinding to a randomly chosen site, with weights
    6N or 6N+1 : move fork
    """

    cdef np.int64_t status
    cdef np.int64_t new_pos, leg_idx, prev_pos, prev_pos2, direction, loop_idx, lattice_increment, chrom_copy

    if event_idx < 4 * system.N_max:
        # Take a step
        if event_idx < 2 * system.N_max:
            leg_idx = event_idx
            direction = -1
        else:
            leg_idx = event_idx - 2 * system.N_max
            direction = 1

        if leg_idx%system.N_max>=system.N:
            #trying to move a LEF that doesn't exist yet!
            print(f"Trying to move {leg_idx} when only {system.N} loop-extruders present!")
            return 0

        if leg_idx >= system.N_max:
            lattice_increment=system.L
        else:
            lattice_increment=0

        prev_pos = system.locs[leg_idx]
        new_pos = system.next_pos(leg_idx, direction) #takes into account circular chromosome and replication.
        chrom_copy = int(new_pos>=system.L)
        
        # check if the loop was attached to the chromatin, and that it's not moving to a fork.
        status = 1
        if (prev_pos >= 0):
            if new_pos%system.L==system.fork[0] or new_pos%system.L==system.fork[1]:
                print("Trying to move to fork! Old pos: ", prev_pos, " new pos: ", new_pos, ". Fork: ", system.fork[0], system.fork[1])
                return 0
            if (new_pos>=system.L) != (prev_pos>=system.L): #if moving between replicated and unreplicated
                print("Trying to move between replicated and unreplicated regions! Old pos: ", prev_pos, " new pos: ", new_pos)
                return 0

            # make a step only if there is no boundary and if not overtaking.
            if (system.perms[prev_pos%system.L + (direction + 1) // 2] > 0) and (system.lattice[chrom_copy, new_pos%system.L+lattice_increment] < 0):
                status *= system.make_step(leg_idx, direction)
                # regenerate events for the previous and the new neighbors, as well as bypassed LEFs
                regenerate_neighbours_and_self(system, evheap, prev_pos)
                regenerate_neighbours_and_self(system, evheap, new_pos)

            # regenerate the performed event
            regenerate_event(system, evheap, event_idx)

            #regenerate unbinding event, now that location is updated
            regenerate_event(system, evheap, 4 * system.N_max + leg_idx)

    elif (event_idx >= 4 * system.N_max) and (event_idx < 5 * system.N_max):
        # unbind the loop
        loop_idx = event_idx - 4 * system.N_max
        if loop_idx>=system.N:
            #trying to move a LEF that doesn't exist yet!
            print(f"Trying to unbind {loop_idx} when only {system.N} loop-extruders present!")
            return 0

        status = 2
        # check if the loop was attached to the chromatin
        if (system.locs[loop_idx] < 0) or (system.locs[loop_idx+system.N_max] < 0):
            status = 0

        # save previous positions, but don't update neighbours until the loop
        # has moved
        prev_pos = system.locs[loop_idx]
        prev_pos2 = system.locs[loop_idx + system.N_max]

        status *= system.detach(loop_idx)

        # regenerate events for the loop itself and for its previous neighbours
        regenerate_all_loop_events(system, evheap, loop_idx)

        # update the neighbours after the loop has been removed
        regenerate_neighbours(system, evheap, prev_pos)
        regenerate_neighbours(system, evheap, prev_pos2)

    elif (event_idx >= 5 * system.N_max) and (event_idx < 6 * system.N_max):
        # a binding event
        loop_idx = event_idx - 5 * system.N_max
        status = 2
        if loop_idx>=system.N:
            #trying to move a LEF that doesn't exist yet!
            print(f"Trying to bind {loop_idx} when only {system.N} loop-extruders present!")
            status=0

        # check if the loop was not attached to the chromatin
        if (system.locs[loop_idx] >= 0) or (system.locs[loop_idx+system.N_max] >= 0):
            print("Trying to bind a loop that is already attached!")
            status = 0

        # find a new position for the LEF (now with weights for each position)
        if system.fork[0]>0:
            #replication started. don't load at forks or unreplicated regions
            new_pos = sample_load_point(system.binding_affinities, np.concatenate((system.locs,system.fork, np.arange(system.fork[1]+system.L, system.fork[0]+system.L+1))))
        else:
            #replication hasn't started. don't load on unreplicated regions
            new_pos = sample_load_point(system.binding_affinities, np.concatenate((system.locs, np.arange(system.L, 2*system.L))))
    
        chrom_copy = int(new_pos>=system.L)

        if (new_pos%system.L==system.fork[0] or new_pos%system.L==system.fork[1]):
            print("binding to fork: ", new_pos)
            print("with fork at: ", system.fork)
            status=0

        if system.lattice[chrom_copy, new_pos%system.L]>=0:
            print("binding to occupied site: ", new_pos)
            print("leg_id=", system.lattice[chrom_copy, new_pos%system.L], " with locs entry ", system.locs[system.lattice[chrom_copy, new_pos%system.L]])
            status=0

        if system.lattice[chrom_copy, new_pos%system.L+system.L]>=0:
            print("binding to occupied site: ", new_pos+system.L)
            print("leg_id=", system.lattice[chrom_copy, new_pos%system.L+system.L], " with locs entry ", system.locs[system.lattice[chrom_copy, new_pos%system.L+system.L]])
            status=0

        # rebind the loop
        status *= system.move_leg(loop_idx, new_pos)
        status *= system.move_leg(loop_idx+system.N_max, new_pos) #note edit; same position. Ok for legs in opposite directions.

        # regenerate events for the loop itself and for its new neighbours
        regenerate_all_loop_events(system, evheap, loop_idx)

        regenerate_neighbours(system, evheap, new_pos)

    elif (event_idx==6*system.N_max or event_idx==6*system.N_max+1):
        status = 4
        if system.fork[0]>system.fork[1]:
            status *= system.move_fork(event_idx-6*system.N_max) #moves fork, potentially updates number of SMCs
            status *= regenerate_after_fork_move(system, evheap, event_idx-6*system.N_max) #unbinds LEFs at fork, updates rates for movement behind and fork move
    else:
        print('event_idx assumed a forbidden value :', event_idx)

    return status

cpdef simulate(p, verbose=False):
    '''Simulate a system of loop extruding LEFs on a 1d lattice.
    Allows to simulate two different types of LEFs, with different
    residence times and rates of backstep.

    Parameters
    ----------
    p : a dictionary with parameters
        PROCESS_NAME : the title of the simulation
        L : the number of sites in the lattice
        N : the number of LEFs
        R_EXTEND : the rate of loop extension,
            can be set globally with a float,
            or individually with an array of floats
        R_SHRINK : the rate of LEF backsteps,
            can be set globally with a float,
            or individually with an array of floats
        R_OFF : the rate of detaching from the polymer,
            can be set globally with a float,
            or individually with an array of floats
        R_ON : the rate of attaching to the polymer,
            can be set globally with a float,
            or individually with an array of floats
        R_BYPASS : rate at which LEFs can jump to same site as another LEF
        REBINDING_TIME : the average time that a LEF spends in solution before
            rebinding to the polymer; can be set globally with a float,
            or individually with an array of floats
        INIT_L_SITES : the initial positions of the left legs of the LEFs,
                       If -1, the position of the LEF is chosen randomly,
                       with both legs next to each other. By default is -1 for
                       all LEFs.
        INIT_R_SITES : the initial positions of the right legs of the LEFs
        ACTIVATION_TIMES : the times at which the LEFs enter the system.
            By default equals 0 for all LEFs.
            Must be 0 for the LEFs with defined INIT_L_SITES
            and INIT_R_SITES.

        T_MAX : the duration of the simulation
        N_SNAPSHOTS : the number of time frames saved in the output. The frames
                      are evenly distributed between 0 and T_MAX.

    '''
    cdef char* PROCESS_NAME = p['PROCESS_NAME']

    cdef np.int64_t L = p['L']
    cdef np.int64_t N = np.round(p['N'])
    cdef np.float64_t T_MAX = p['T_MAX']
    cdef np.int64_t N_SNAPSHOTS = p['N_SNAPSHOTS']
    cdef np.float64_t BYPASS_RATE = p['R_BYPASS']
    cdef np.float64_t BURNIN_TIME = p.get('BURNIN_TIME', 0)
    cdef np.float64_t FORK_RATE = p.get('R_FORK', 0.) #note that replication is only started after a burn in period
    cdef np.int64_t N_MAX = N*2

    cdef np.int64_t i

    cdef np.float64_t [:] VELS = np.zeros(4*N_MAX, dtype=np.float64)
    cdef np.float64_t [:] UNBINDING_RATES = np.zeros(2*L, dtype=np.float64)
    cdef np.float64_t [:] BINDING_RATES = np.ones(2*L, dtype=np.float64)
    cdef np.float64_t [:] REBINDING_TIMES = np.zeros(N_MAX, dtype=np.float64)

    for i in range(N_MAX):
        VELS[i] =   VELS[i+3*N_MAX] = p['R_EXTEND'][i] if type(p['R_EXTEND']) in (list, np.ndarray) else p['R_EXTEND']
        VELS[i+N] = VELS[i+2*N_MAX] = p['R_SHRINK'][i] if type(p['R_SHRINK']) in (list, np.ndarray) else p['R_SHRINK']
        REBINDING_TIMES[i] = (
            (p['REBINDING_TIME'][i])
            if type(p.get('REBINDING_TIME',0)) in (list, np.ndarray)
            else p.get('REBINDING_TIME',0))

    for i in range(2*L):
        UNBINDING_RATES[i] = p['R_OFF'][i] if type(p['R_OFF']) in (list, np.ndarray) else p['R_OFF']
        BINDING_RATES[i] = p['R_ON'][i] if type(p['R_ON']) in (list, np.ndarray) else p['R_ON']
    #the system class makes sure binding rates are zero on unreplicated regions

    cdef np.int64_t [:] INIT_LOCS = (-1) * np.ones(2*N, dtype=np.int64)
    if ('INIT_L_SITES' in p) and ('INIT_R_SITES' in p):
        for i in range(N):
            INIT_LOCS[i] = p['INIT_L_SITES'][i]
            INIT_LOCS[i+N] = p['INIT_R_SITES'][i]

    cdef np.float64_t [:] ACTIVATION_TIMES = p.get('ACTIVATION_TIMES',
        np.zeros(N, dtype=np.float64))

    if (not (INIT_LOCS is None)) and (INIT_LOCS.size != 2*N):
        raise Exception(
            'The length of the provided array of initial positions should be 2N')

    for i in range(N):
        if INIT_LOCS[i] != -1:
            assert (INIT_LOCS[i+N] != -1)
            assert ACTIVATION_TIMES[i] == 0
        else:
            assert (INIT_LOCS[i+N] == -1)

    cdef np.float_t [:] PERMS = p.get('PERMS', None)
    if (not (PERMS is None)) and (PERMS.size != L+1):
        raise Exception(
            'The length of the provided array of permeabilities should be L+1')

    cdef System system = System(L, N, VELS, REBINDING_TIMES, UNBINDING_RATES, BYPASS_RATE, INIT_LOCS, PERMS, BINDING_RATES)

    cdef np.int64_t [:,:] l_sites_traj = np.zeros((N_SNAPSHOTS, N_MAX), dtype=np.int64)
    cdef np.int64_t [:,:] r_sites_traj = np.zeros((N_SNAPSHOTS, N_MAX), dtype=np.int64)
    cdef np.float64_t [:] ts_traj = np.zeros(N_SNAPSHOTS, dtype=np.float64)
    cdef np.int64_t [:,:] fork_traj = np.zeros((N_SNAPSHOTS, 2), dtype=np.int64)

    cdef np.int64_t last_event = 0

    cdef np.float64_t prev_snapshot_t = 0
    cdef np.float64_t tot_rate = 0
    cdef np.int64_t snapshot_idx = 0

    cdef Event_heap evheap = Event_heap()

    # Move LEFs onto the lattice at the corresponding activations times.
    # If the positions were predefined, initialize the fall-off time in the
    # standard way.
    for i in range(system.N): #only activate the first N
        # if the loop location is not predefined, activate it
        # at the predetermined time
        if (INIT_LOCS[i] == -1) and (INIT_LOCS[i] == -1):
            evheap.add_event(i + 5 * system.N_max, ACTIVATION_TIMES[i])

        # otherwise, the loop is already placed on the lattice and we need to
        # regenerate all of its events and the motion of its neighbours
        else:
            regenerate_all_loop_events(system, evheap, i)
            regenerate_neighbours(system, evheap, INIT_LOCS[i])
            regenerate_neighbours(system, evheap, INIT_LOCS[i+system.N])

    cdef Event_t event
    cdef np.int64_t event_idx

    cdef np.float64_t t=0
    #Burn in the configuration
    if verbose:
        print(PROCESS_NAME, 'burning in...')

    while t < BURNIN_TIME:
        event = evheap.pop_event()
        system.time = event.time
        event_idx = event.event_idx
        status = do_event(system, evheap, event_idx)

        if status == 0:
            print('an assertion failed somewhere')
            return 0
        t = event.time

    if verbose:
        print(PROCESS_NAME, 'Burn-in complete. Collecting snapshots and starting replication.')

    start_replication(system, evheap, FORK_RATE)

    while snapshot_idx < N_SNAPSHOTS:
        event = evheap.pop_event()
        system.time = event.time
        event_idx = event.event_idx

        status = do_event(system, evheap, event_idx)

        if status == 0:
            print('an assertion failed somewhere')
            return 0

        if system.time-BURNIN_TIME > prev_snapshot_t + T_MAX / N_SNAPSHOTS:
            prev_snapshot_t = system.time-BURNIN_TIME
            l_sites_traj[snapshot_idx] = system.locs[:N_MAX]
            r_sites_traj[snapshot_idx] = system.locs[N_MAX:]
            fork_traj[snapshot_idx]=system.fork
            ts_traj[snapshot_idx] = system.time-BURNIN_TIME
            snapshot_idx += 1
            if verbose and (snapshot_idx % 10 == 0):
                print(PROCESS_NAME, snapshot_idx, system.time-BURNIN_TIME, T_MAX)
            np.random.seed()

    return np.array(l_sites_traj), np.array(r_sites_traj), np.array(fork_traj), np.array(ts_traj)
