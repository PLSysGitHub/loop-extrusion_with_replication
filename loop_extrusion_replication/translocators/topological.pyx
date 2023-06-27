#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True

#This script has been edited so that each SMC
#consists of two legs that have upto 4 strands of DNA in them
#This also means that the direction for each strand in each leg has to be specified

#By-passing of topological loop-extruders is not allowed


import numpy as np
cimport numpy as np 
import cython
cimport cython 


cdef extern from "<stdlib.h>":
    double drand48()   

cdef cython.double randnum():
    return drand48()


cdef class topologicalSmcForkTranslocator(object):
    cdef int N #length one chromosome
    cdef int n #length lin segment
    cdef int M #current number loop extruders
    cdef int smc_per_chrom
    cdef int fork1 #fork positions
    cdef int fork2
    cdef cython.double [:] emission
    cdef cython.double [:] stallLeft
    cdef cython.double [:] stallRight
    cdef cython.double [:] stallFalloff
    cdef cython.double [:] falloff
    cdef cython.double [:] pauseL
    cdef cython.double [:] pauseR
    cdef cython.double [:] cumEmission
    cdef cython.long [:,:] SMCs1
    cdef cython.long [:,:] SMCs2
    cdef cython.long [:,:] dir1
    cdef cython.long [:,:] dir2
    cdef cython.long [:] stalled1 
    cdef cython.long [:] stalled2
    cdef cython.long [:,:] occupant # index of occupant, which leg of occupant
    cdef int maxss
    cdef int curss
    cdef cython.long [:] ssarray  
    cdef cython.double kForkMoves
    cdef cython.double stallFork
    cdef cython.double knockOffProb_OriToTer
    cdef cython.double knockOffProb_TerToOri
    cdef cython.bint replication_started
    
    def __init__(self, fork_start, emissionProb, deathProb, stallProbLeft, stallProbRight, pauseProbL, pauseProbR, stallFalloffProb, kkOriToTer_kkTerToOri, numSmc, forkMoveRate, stallFork):
     
        self.N = len(emissionProb)//2
        self.replication_started=0
        self.fork1=fork_start
        self.fork2=self.N-1-fork_start
        self.n=2*fork_start

        self.M = numSmc
        self.smc_per_chrom=numSmc
        self.emission = emissionProb
        self.knockOffProb_OriToTer = kkOriToTer_kkTerToOri[0]
        self.knockOffProb_TerToOri = kkOriToTer_kkTerToOri[1] # rate of facilitated dissociation
        self.stallLeft = stallProbLeft
        self.stallRight = stallProbRight
        self.falloff = deathProb
        self.pauseL = pauseProbL
        self.pauseR = pauseProbR
        cumem = np.cumsum(emissionProb)
        cumem = cumem / float(cumem[len(cumem)-1])
        self.cumEmission = np.array(cumem, np.double)

        self.SMCs1 = np.zeros(((self.M*2), 2), int) #arrays two times larger because final state has 2 times more smcs and monomers
        self.SMCs2 = np.zeros(((self.M*2), 2), int) #second index runs over strands in condensin leg

        self.dir1 = -1*np.ones(((self.M*2),2),int)
        self.dir2= np.ones(((self.M*2),2),int)

        self.stalled1 = np.zeros(self.M*2, int)
        self.stalled2 = np.zeros(self.M*2, int)
        self.occupant = -1*np.ones((self.N*2,2), int)#if negative, not occupied. otherwise has index of occupant and which leg of occupant

        self.stallFalloff = stallFalloffProb
        self.stallFork=stallFork
        self.kForkMoves=forkMoveRate

        self.emission[self.N:2*self.N]=0. #don't load on unreplicated
        
        self.maxss = 100 #lower because need to redraw as polymer is replicated
        self.curss = self.maxss+1

        for ind in xrange(self.M):
            self.birth(ind)

    cdef calcInd(self, i):
        #Helper function that returns a proper index for statements like i+1
        #Takes care of periodic BCs and forks
        if i<self.N:
            return np.mod(i,self.N)
        elif i==self.fork1+self.N:
            return self.fork1
        elif i==self.fork2+self.N:
            return self.fork2
        else:
            return np.mod(i,self.N)+self.N

    cdef birth(self, cython.int ind):
        cdef int pos,i 
  
        while True:
            pos = self.getss()
            if pos>self.N*2-1:
                print "out of bounds value", pos, self.cumEmission[self.N*2-1]
                continue  
            if (pos >=self.fork1+self.N and pos<=self.fork2+self.N): 
                print "unreplicated value", pos, self.cumEmission[len(self.cumEmission)-1]
                print "with forks at", self.fork1,self.fork2
                continue 
            if pos < 0: 
                print "negative value", pos, self.cumEmission[0]
                continue 
 
            sec_pos = self.calcInd(pos+1)#periodic bcs for both copies of chromosome

            if (self.occupant[pos,0] >= 0 or self.occupant[sec_pos,0]>=0):#note edit; non-negative are indices of occupants
                continue #don't load where occupied
            
            self.SMCs1[ind,:] = pos
            self.dir1[ind,:] =-1 #because lower than sec_pos
            self.SMCs2[ind,:] = sec_pos
            self.dir2[ind,:] = 1
            
            #update occupant array
            self.occupant[pos,0] =ind
            self.occupant[pos,1]=1
            self.occupant[sec_pos,0]=ind
            self.occupant[sec_pos,1] = 2

            return

    cdef death(self):
        cdef int i 
        cdef double falloff1, falloff2 
        cdef double falloff 
         
        for i in xrange(self.M):
            in1=self.SMCs1[i,:]
            in2=self.SMCs2[i,:]

            if self.stalled1[i] == 0:
                falloff1 = max(self.falloff[in1[0]], self.falloff[in1[1]]) #at fork this is higher
            else: 
                #edited: max, near ter stalling can otherwise decrease falloff
                falloff1 = max(self.stallFalloff[in1[0]],self.falloff[in1[0]],self.stallFalloff[in1[1]],self.falloff[in1[1]])

            if self.stalled2[i] == 0:
                falloff2 = max(self.falloff[in2[0]], self.falloff[in2[1]]) #at fork this is higher
            else:
                #edited: max, near ter stalling can otherwise decrease falloff
                falloff2 = max(self.stallFalloff[in2[0]],self.falloff[in2[0]],self.stallFalloff[in2[1]],self.falloff[in2[1]])
            
            falloff = max(falloff1, falloff2)
            if randnum() < falloff:                 
                self.occupant[in1[0],0] =-1
                self.occupant[in1[1],0] =-1
                self.occupant[in2[0],0] =-1
                self.occupant[in2[1],0] =-1
                self.stalled1[i] = 0
                self.stalled2[i] = 0
                self.birth(i)
    
    cdef int getss(self):
    
        if self.curss >= self.maxss - 1:
            foundArray = np.array(np.searchsorted(self.cumEmission, np.random.random(self.maxss)), dtype = np.long)
            self.ssarray = foundArray
            self.curss = -1
        
        self.curss += 1         
        return self.ssarray[self.curss]
        
    cdef next_positions(self, pos, ds):
        #pos is now a vector of two strands
        #ds is now a vector of two directions +/- 1
        cdef long [:] newPos
        cdef long [:] newDs
    
        if pos[0]==pos[1]: #single strand behaviour
            cur=pos[0]
            d=ds[0]
            nxt=self.calcInd(cur+d)

            #Check we're not at a crazy index
            if (cur>=self.fork1+self.N and cur<=self.fork2+self.N):
                print("\n\ncur is on unreplicated locus!\n\n")
            
            if cur<self.N:
                if nxt ==self.fork1 and self.fork1>0: 
                    newPos =np.array([nxt, self.fork1+self.N-1])
                    newDs = np.array([d,-1])
                elif nxt==self.fork2 and self.fork2<self.N-1:
                    newPos = np.array([nxt, self.fork2+self.N+1])
                    newDs =np.array([d,1])
                else:
                    newPos= np.array([nxt,nxt])
                    newDs = ds
            else:
                if nxt==self.fork1:#calcInd returns fork (no +N!) if reach it from newly replicated chrom
                    newPos =np.array([nxt, self.calcInd(nxt-1)])
                    newDs =np.array([1,-1])
                elif nxt==self.fork2: 
                    newPos =np.array([nxt, self.calcInd(nxt+1)])
                    newDs = np.array([-1,1])
                else:
                    newPos= np.array([nxt,nxt])
                    newDs = ds

        else: #two separate strands
            nxt1=self.calcInd(pos[0]+ds[0])
            nxt2=self.calcInd(pos[1]+ds[1])
            #note that rejoining only happens when leg has gone across replicated segments
            # in this case, calcInd returns a fork position for nxt2
            if nxt2 != self.fork1 and nxt2 != self.fork2: #not joining
                newPos= np.array([nxt1,nxt2])
                newDs = ds
            else: #join
                newPos=np.array([nxt1,nxt1])
                newDs =np.array([ds[0],ds[0]])

        return newPos, newDs

    cdef step(self):
        cdef long i 
        cdef double pause
        cdef long [:] cur1
        cdef long [:] cur2 
        cdef long [:] newCur1
        cdef long [:] newCur2
        for i in range(self.M):            
            # update positions             
            cur1 = self.SMCs1[i, :]
            d1= self.dir1[i,:]
            cur2 = self.SMCs2[i, :]
            d2= self.dir2[i,:]
            
            # new positions taking into account topological behaviour at forks
            newCur1,newD1= self.next_positions(cur1,d1)
            newCur2, newD2 = self.next_positions(cur2,d2)

            # reset "is stalled" -> this actually means "will dissociate from collision"
            self.stalled1[i] = 0
            self.stalled2[i] = 0               
            
            # move each side only if the other side is "free" or stall/knock off
            
            #First leg
            pause1 = max(self.pauseL[cur1[0]], self.pauseL[cur1[2]])
            if randnum() > pause1: 
                if (self.occupant[newCur1[0],0] < 0 and self.occupant[newCur1[1],0] < 0):#new site unoccupied
                    # take a normal step 
                    self.occupant[cur1[0],0] = -1 #old position no longer occupied
                    self.occupant[cur1[1],0] = -1 #old position no longer occupied
                    self.occupant[newCur1[0],0] =i
                    self.occupant[newCur1[1],0] =i
                    self.occupant[newCur1[0],1] =1
                    self.occupant[newCur1[1],1] =1
                    self.SMCs1[i,0] = newCur1[0]
                    self.dir1[i,0]=newD1[0]
                    self.SMCs1[i,1] = newCur1[1]
                    self.dir1[i,1]=newD1[1]
                else:
                    if randnum() <= self.knockOffProb_OriToTer:
                        #stall and maybe dissociate
                        self.stalled1[i] = 1
                        #note no by-passing for topological loop-extruders!

            #Other leg
            pause2 = max(self.pauseR[cur2[0]], self.pauseR[cur2[1]])
            if randnum() > pause2:                 
                if (self.occupant[newCur2[0],0] < 0 and self.occupant[newCur2[1],0]<0) :        
                    # take a normal step 
                    self.occupant[cur2[0],0] = -1 #old position no longer occupied
                    self.occupant[cur2[1],0] = -1 #old position no longer occupied
                    self.occupant[newCur2[0],0] = i
                    self.occupant[newCur2[0],1] = 2
                    self.occupant[newCur2[1],0] = i
                    self.occupant[newCur2[1],1] = 2
                    self.SMCs2[i,0] = newCur2[0]
                    self.dir2[i,0]=newD2[0]
                    self.SMCs2[i,1] = newCur2[1]
                    self.dir2[i,1]=newD2[1]
                else:
                    if randnum() <= self.knockOffProb_TerToOri:
                        # stall and maybe dissociate
                        self.stalled2[i] = 1

            
            # mark for dissociation if either side is stalled      
            if  (self.stalled2[i] == 1) or (self.stalled1[i] == 1):
                self.stalled1[i] = 1
                self.stalled2[i] = 1             
    
    def update_cumEmission(self):
        cumem=np.zeros(2*self.N)
        x=0
        
        for i in range(2*self.N):
            x+=self.emission[i]
            cumem[i]=x
        cumem = cumem / x
        self.cumEmission = cumem

        #need to redraw loading positions
        self.curss=self.maxss+1
    
    def move_fork(self):
        fork_moved=False
        if self.fork1<self.N//2 and randnum()<=self.kForkMoves:

            #reset to what they were before fork there
            self.falloff[self.fork1]=self.falloff[self.fork1+self.N]
            if self.fork1>0:
                self.emission[self.N+self.fork1-1]=self.emission[self.fork1-1] #note -1; otherwise load (fork1-1+N, fork1+N)!
            
            #move fork
            self.fork1+=1

            #for condensin leg at fork, maybe split
            at_fork, which_leg=self.occupant[self.fork1]
            if at_fork >=0:
                #is an index
                if which_leg==1:
                    if self.SMCs1[at_fork,0]==self.SMCs1[at_fork,1]: #split
                        self.SMCs1[at_fork,1]+=self.N
                        #directions remain same
                    else: #rejoin
                        self.SMCs1[at_fork,1]=self.SMCs1[at_fork,0]
                        self.dir1[at_fork,1]=self.dir1[at_fork,0]
                else:
                    if self.SMCs2[at_fork,0]==self.SMCs2[at_fork,1]: #split
                        self.SMCs2[at_fork,1]+=self.N
                        #directions remain same
                    else: #rejoin
                        self.SMCs2[at_fork,1]=self.SMCs2[at_fork,0]
                        self.dir2[at_fork,1]=self.dir2[at_fork,0]

            
            #at fork, increase falloff
            self.falloff[self.fork1]=max(self.stallFork,self.falloff[self.fork1])
            fork_moved=True
            
        if self.fork2>self.N//2 and randnum()<=self.kForkMoves:
            #reset to what they were before fork there
            self.falloff[self.fork2]=self.falloff[self.fork2+self.N]
            self.emission[self.N+self.fork2]=self.emission[self.fork2] #then can load at fork2+1 and fork2+2 
           
            
            #move fork
            self.fork2-=1
            
            #for condensin leg at fork, do split
            at_fork, which_leg=self.occupant[self.fork2]
            if at_fork >=0:#is an index
                if which_leg==1:
                    if self.SMCs1[at_fork,0]==self.SMCs1[at_fork,1]: #split
                        self.SMCs1[at_fork,1]+=self.N
                        #directions remain same
                    else: #rejoin
                        self.SMCs1[at_fork,1]=self.SMCs1[at_fork,0]
                        self.dir1[at_fork,1]=self.dir1[at_fork,0]
                else:
                    if self.SMCs2[at_fork,0]==self.SMCs2[at_fork,1]: #split
                        self.SMCs2[at_fork,1]+=self.N
                        #directions remain same
                    else: #rejoin
                        self.SMCs2[at_fork,1]=self.SMCs2[at_fork,0]
                        self.dir2[at_fork,1]=self.dir2[at_fork,0]
            
            #at fork, increase falloff
            self.falloff[self.fork2]=max(self.stallFork,self.falloff[self.fork2])

            fork_moved=True

        if fork_moved:
            self.update_cumEmission() #can load on newly replicated bits
            self.n=self.fork1+(self.N-1-self.fork2)
            new_M=round(self.smc_per_chrom*(self.N+self.n)/self.N) #number smcs proportional to chrom length
            added=new_M-self.M
            for i in range(added):
                self.M+=1
                self.birth(self.M-1) #load the new condensin somewhere      
            #check if replication is complete
            if self.fork1>=self.N//2 and self.fork2<=self.N//2:
                self.replication_started=0

    def steps(self,N):
        cdef int i 
        for i in xrange(N):
            self.death()
            self.step()
            if self.replication_started: #replication ongoing
                self.move_fork()
            
    def getOccupied(self):
        return np.array(self.occupant)
    
    def getSMCs(self): #note edit; only SMCs actually in use
        return np.array(self.SMCs1[0:self.M,:]), np.array(self.SMCs2[0:self.M,:])
    
    def getPairwiseSMCs(self): 
        #for compatibility with 3D code
        # for each condensin with x strands in 1st leg and y strands in second leg
        # return links between all pairs in 1, all pairs in 2, and all pairs between 1,2
        
        legs1=[]
        legs2=[]

        for i in range(self.M):
            in_1=np.array(self.SMCs1[i,:])
            in_2=np.array(self.SMCs2[i,:])
            both=np.unique(np.concatenate((in_1,in_2), axis=0))
            for i in range(both.size):
                for j in range(i+1,both.size):
                    legs1.append(both[i])
                    legs2.append(both[j])

        return np.array(legs1), np.array(legs2)


    def getFork(self):
        return np.array([self.fork1, self.fork2])
    
    def getN(self):
        return self.N

    def getM(self):
        return self.M

    def start_replication(self):
        print "\nStarting replication\n"
        self.replication_started=1

