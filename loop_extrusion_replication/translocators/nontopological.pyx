#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True

import numpy as np
cimport numpy as np 
import cython
cimport cython 


cdef extern from "<stdlib.h>":
    double drand48()   

cdef cython.double randnum():
    return drand48()


cdef class smcForkTranslocator(object):
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
    cdef cython.long [:] SMCs1
    cdef cython.long [:] SMCs2
    cdef cython.long [:] dir1
    cdef cython.long [:] dir2
    cdef cython.long [:] stalled1 
    cdef cython.long [:] stalled2
    cdef cython.long [:] occupied 
    
    cdef int maxss
    cdef int curss
    cdef cython.double knockOffProb_TerToOri
    cdef cython.double knockOffProb_OriToTer
    cdef cython.double kBypass
    cdef cython.long [:] ssarray  
    cdef cython.double kForkMoves
    cdef cython.double stallFork
    cdef cython.bint replication_started
    
    def __init__(self, fork_start, emissionProb, deathProb, stallProbLeft, stallProbRight, pauseProbL, pauseProbR, stallFalloffProb, kkOriToTer_kkTerToOri_kBypass,  numSmc, forkMoveRate, stallFork):
     
        self.N = len(emissionProb)//2
        
        self.replication_started=0
        self.fork1=fork_start
        self.fork2=self.N-1-fork_start
        self.n=2*fork_start

        self.M = numSmc
        self.smc_per_chrom=numSmc
        self.emission = emissionProb
        self.stallLeft = stallProbLeft
        self.stallRight = stallProbRight
        self.falloff = deathProb
        self.pauseL = pauseProbL
        self.pauseR = pauseProbR
        cumem = np.cumsum(emissionProb)
        cumem = cumem / float(cumem[len(cumem)-1])
        self.cumEmission = np.array(cumem, np.double)
        self.SMCs1 = np.zeros((self.M*2), int) #arrays two times larger because final state has 2 times more smcs and monomers
        self.SMCs2 = np.zeros((self.M*2), int)
        self.dir1 = -1*np.ones((self.M*2),int)
        self.dir2= np.ones((self.M*2),int)
        self.stalled1 = np.zeros(self.M*2, int)
        self.stalled2 = np.zeros(self.M*2, int)
        self.occupied = np.zeros(self.N*2, int)
        self.stallFalloff = stallFalloffProb
        self.stallFork=stallFork
        self.kForkMoves=forkMoveRate

        self.emission[self.N+fork_start:2*self.N-fork_start]=0. #don't load on unreplicated
        
        self.knockOffProb_OriToTer = kkOriToTer_kkTerToOri_kBypass[0]
        self.knockOffProb_TerToOri = kkOriToTer_kkTerToOri_kBypass[1] # rate of facilitated dissociation
        self.kBypass = kkOriToTer_kkTerToOri_kBypass[2]
        
        self.maxss = 100 #lower because need to redraw as polymer is replicated
        self.curss = self.maxss+1

        for ind in xrange(self.M):
            self.birth(ind)


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
 
            if (self.occupied[pos] >= 1):
                continue #don't load where occupied
            self.SMCs1[ind] = pos
            self.SMCs2[ind] = pos
            self.dir1[ind] = -1
            self.dir2[ind] = 1
            self.occupied[pos] += 2 
            
            return

    cdef death(self):
        cdef int i 
        cdef double falloff1, falloff2 
        cdef double falloff 
         
        for i in xrange(self.M):
            if self.stalled1[i] == 0:
                falloff1 = self.falloff[self.SMCs1[i]] #at fork this is higher
            else: 
                #edited: max, near ter stalling can otherwise decrease falloff
                falloff1 = max(self.stallFalloff[self.SMCs1[i]],self.falloff[self.SMCs1[i]])

            if self.stalled2[i] == 0:
                falloff2 = self.falloff[self.SMCs2[i]]
            else:
                #edited: max, near ter stalling can otherwise decrease falloff
                falloff2 = max(self.stallFalloff[self.SMCs2[i]],self.falloff[self.SMCs2[i]])
            
            falloff = max(falloff1, falloff2)
            if randnum() < falloff:                 
                self.occupied[self.SMCs1[i]] -= 1
                self.occupied[self.SMCs2[i]] -= 1
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
        
    cdef next_position(self, cur,d):
        #calculate where condensin legs should go

        if cur<self.N:
            nxt=np.mod(cur+d,self.N)
            if nxt ==self.fork1 and self.fork1>0: #approach fork
                if randnum()<0.5: #50% chance to keep going
                    next1=self.fork1
                    nd1=d
                else: #50% change to hop to newly replicated
                    next1=np.mod(self.fork1-1,self.N)+self.N
                    nd1= -1
            elif nxt==self.fork2 and self.fork2<self.N-1: #approach other fork
                if randnum()<0.5: #keep going
                    next1=self.fork2
                    nd1=d
                else: #switch to newly replicated
                    next1=np.mod(self.fork2+1,self.N)+self.N
                    nd1= 1 
            else:
                next1= nxt
                nd1=d
        else:
            nxt=np.mod(cur+d, self.N)+self.N
            if nxt==self.fork2+self.N:
                next1=self.fork2
                if randnum()<0.5: #50% chance to change direction
                    nd1=1
                else:
                    nd1=-1
            elif nxt==self.fork1+self.N:
                next1=self.fork1
                if randnum()<0.5: #50% chance to change direction
                    nd1=1
                else:
                    nd1=-1
            elif cur>=self.fork1+self.N and cur<=self.fork2+self.N:
                print "with forks at", self.fork1,self.fork2
                raise Exception("cur is on unreplicated locus ",cur)
            else:
                next1= nxt
                nd1=d

        return next1,nd1

    cdef step(self):
        cdef int i 
        cdef double pause
        cdef int cur1
        cdef int cur2 
        cdef int newCur1
        cdef int newCur2
        for i in range(self.M):            
            # update positions             
            cur1 = self.SMCs1[i]
            cur2 = self.SMCs2[i]
            d1=self.dir1[i]
            d2=self.dir2[i]
            # new positions with the periodic boundaries
            newCur1,newD1=self.next_position(cur1,d1)
            newCur2, newD2=self.next_position(cur2,d2)

            # reset "is stalled" -> this actually means "will dissociate from collision"
            self.stalled1[i] = 0
            self.stalled2[i] = 0               
            
            # move each side only if the other side is "free" or stall/knock off
            
            #First leg
            pause1 = self.pauseL[self.SMCs1[i]]
            if randnum() > pause1: 
                if (self.occupied[newCur1] == 0):
                    # take a normal step 
                    self.occupied[newCur1] += 1
                    self.occupied[cur1] -= 1
                    self.SMCs1[i] = newCur1 
                    self.dir1[i] = newD1
                else:
                    rateSum = self.knockOffProb_OriToTer+self.kBypass
                    if randnum() <= rateSum: 
                        if randnum() <= self.knockOffProb_OriToTer/rateSum:
                            # stall and maybe dissociate
                            self.stalled1[i] = 1
                        else:
                            self.occupied[newCur1] += 1
                            self.occupied[cur1] -= 1
                            self.SMCs1[i] = newCur1
                            self.dir1[i] = newD1
                        
            #Other leg
            pause2 = self.pauseR[self.SMCs2[i]]
            if randnum() > pause2:                 
                if (self.occupied[newCur2] == 0) :        
                    # take a normal step 
                    self.occupied[newCur2] += 1
                    self.occupied[cur2] -= 1
                    self.SMCs2[i] = newCur2
                    self.dir2[i] = newD2
                else:
                    rateSum = self.knockOffProb_TerToOri+self.kBypass
                    if randnum() <= rateSum:                     
                        if randnum() <= self.knockOffProb_TerToOri/rateSum:
                            # stall and maybe dissociate
                            self.stalled2[i] = 1     
                        else:
                            # bypass
                            self.occupied[newCur2] += 1
                            self.occupied[cur2] -= 1
                            self.SMCs2[i] = newCur2
                            self.dir2[i] = newD2
                    
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
            self.emission[self.N+self.fork1]=self.emission[self.fork1] #emission zero from fork1 to fork2
            
            #for each condensin leg at fork, choose which strand it goes on
            if self.occupied[self.fork1]>0:
                for i in range(self.M):#note directions don't change!
                    if self.SMCs1[i]==self.fork1 and randnum()<0.5:
                        self.SMCs1[i]+=self.N
                    if self.SMCs2[i]==self.fork1 and randnum()<0.5:
                        self.SMCs2[i]+=self.N

            #move fork
            self.fork1+=1
            
            #at fork, increase falloff
            self.falloff[self.fork1]=max(self.stallFork,self.falloff[self.fork1])
            fork_moved=True
            
        if self.fork2>self.N//2 and randnum()<=self.kForkMoves:
            #reset to what they were before fork there
            self.falloff[self.fork2]=self.falloff[self.fork2+self.N]
            self.emission[self.N+self.fork2]=self.emission[self.fork2] 
           
            #for each condensin leg at fork, choose which strand it goes on
            if self.occupied[self.fork2]>0:
                for i in range(self.M): #directions don't change!
                    if self.SMCs1[i]==self.fork2 and randnum()<0.5:
                        self.SMCs1[i]+=self.N
                    if self.SMCs2[i]==self.fork2 and randnum()<0.5:
                        self.SMCs2[i]+=self.N
            
            #move fork
            self.fork2-=1
            
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
        return np.array(self.occupied)
    
    def getSMCs(self): #note edit; only SMCs actually in use
        return np.array(self.SMCs1)[0:self.M], np.array(self.SMCs2)[0:self.M]
    
    def getFork(self):
        return np.array([self.fork1, self.fork2])
    
    def getN(self):
        return self.N

    def getM(self):
        return self.M

    def updateMap(self, cmap):
        cmap[self.SMCs1, self.SMCs2] += 1
        cmap[self.SMCs2, self.SMCs1] += 1

    def updatePos(self, pos, ind):
        pos[ind, self.SMCs1] = 1
        pos[ind, self.SMCs2] = 1

    def start_replication(self):
        print "\nStarting replication\n"
        self.replication_started=1

