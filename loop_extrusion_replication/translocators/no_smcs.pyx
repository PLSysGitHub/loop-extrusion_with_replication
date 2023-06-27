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


cdef class forkTranslocator(object):
    cdef int N #length one chromosome
    cdef int n #length lin segment
    cdef int fork1 #fork positions
    cdef int fork2
    cdef cython.double kForkMoves
    cdef cython.bint replication_started
    
    def __init__(self,N, fork_start, forkMoveRate):
     
        self.N = N
        
        self.replication_started=0
        self.fork1=fork_start
        self.fork2=self.N-1-fork_start
        self.n=2*fork_start

        self.kForkMoves=forkMoveRate
    
    def move_fork(self):
        fork_moved=False
        if self.fork1<self.N//2 and randnum()<=self.kForkMoves:
            #move fork
            self.fork1+=1
            
            fork_moved=True
            
        if self.fork2>self.N//2 and randnum()<=self.kForkMoves:
            #move fork
            self.fork2-=1

            fork_moved=True

        if fork_moved:
            self.n=self.fork1+(self.N-1-self.fork2)
            if self.fork1>=self.N//2 and self.fork2<=self.N//2:
                self.replication_started=0

    def steps(self,N):
        cdef int i 
        for i in xrange(N):
            if self.replication_started: #replication ongoing
                self.move_fork()
            
    
    def getFork(self):
        return np.array([self.fork1, self.fork2])
    
    def getN(self):
        return self.N


    def start_replication(self):
        print "\nStarting replication\n"
        self.replication_started=1

