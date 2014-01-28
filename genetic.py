#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Moto: Induction motor parameter estimation tool

Genetic Algorithms

Author: Julius Susanto
Last edited: January 2014
"""
import numpy as np
from common_calcs import *

"""
GA_SOLVER  - Genetic algorithm solver for double cage model with core losses
             Solves for 6 circuit parameters [Xs Xm Rr1 Xr1 Rr2 Rc]
             Rs and Xr2 are computed by linear restrictions
             Includes change of variables
             Includes adaptive step size (as per Pedra 2008)
             Includes determinant check of jacobian matrix

Usage: nr_solver (p, pop, n_r, n_e, c_f, n_gen, err_tol)

Where   p is a vector of motor performance parameters:
        p = [sf eff pf Tb Tlr Ilr]
          sf = full-load slip
          eff = full-load efficiency
          pf = full-load power factor
          T_b = breakdown torque (as # of FL torque)
          T_lr = locked rotor torque (as # of FL torque)
          I_lr = locked rotor current
        pop is the population each generation
        n_r is the number of members retained for mating
        n_e is the number of elite children
        c_f is the crossover fraction
        n_gen is the maximum number of generations  
        err_tol is the error tolerance for convergence

Returns:   x is a vector of motor equivalent parameters:
          x = [Rs Xs Xm Rr1 Xr1 Rr2 Xr2 Rc]
           x(0) = Rs = stator resistance
           x(1) = Xs = stator reactance
           x(2) = Xm = magnetising reactance
           x(3) = Rr1 = rotor / inner cage resistance
           x(4) = Xr1 = rotor / inner cage reactance
           x(5) = Rr2 = outer cage resistance
           x(6) = Xr2 = outer cage reactance
           x(7) = Rc = core resistance
          iter is the number of iterations
          err is the squared error of the objective function
          conv is a true/false flag indicating convergence
"""    
def ga_solver(self, p, pop, n_r, n_e, c_f, n_gen, err_tol):
    
    # Standard deviation weighting vector for mutation noise
    sigma = [0.01, 0.01, 0.33, 0.01, 0.01, 0.01, 0.01, 6.67]; 
    
    # Initial weighting vector
    w = np.matrix([0.15, 0.15, 5.0, 0.15, 0.3, 0.15, 0.15, 100.0])  
    
    # Human-readable motor performance parameters
    # And base value initialisation
    sf = p[0]                          # Full-load slip (pu)
    eff = p[1]                         # Full-load efficiency (pu)
    pf = p[2]                          # Full-load power factor (pu)
    T_fl = pf * eff / (1 - sf)         # Full-load torque (pu)
    T_b = p[3] * T_fl                  # Breakdown torque (pu)
    T_lr = p[4] * T_fl                 # Locked rotor torque (pu)
    i_lr = p[5]                        # Locked rotor current (pu)
    Pm_fl = pf * eff                   # Mechanical power (at FL)
    Q_fl = np.sin(np.arccos(pf))         # Full-load reactive power (pu)
    
    # Formulate solution
    pqt = [Pm_fl, Q_fl, T_b, T_lr, i_lr, eff]
    
    # Initial settings
    gen = 1
    conv = 0
    
    # Create initial population
    wmat = np.dot(w.transpose(), np.matrix(np.ones(pop)))
    x = np.array(wmat.transpose()) * np.random.rand(pop,8)
    err = np.zeros(pop)
    
    # Check solution of initial population
    for i in range(0,pop):
        diff = np.subtract(pqt, calc_pqt(sf,x[i,:]))
        y = np.divide(diff, pqt)
        err[i] = np.dot(y, np.transpose(y))
        
        if err[i] < err_tol:
            z = x[i,:]
            conv = 1
            return z, gen, err[i], conv
    
    # Run genetic algorithm
    for gen in range(2,n_gen+1):
        
        self.statusBar().showMessage('Calculating generation %d...' % gen)
        
        # Select for fitness
        fitness = np.sort(err)
        index = np.argsort(err)
        
        # Create next generation
        x_new = np.zeros((pop,8))
        x_mate = x[index[0:n_r],:]         # select mating pool
        
        # Elite children (select best "n_e" children for next generation)
        x_new[0:n_e,:] = x[index[0:n_e],:]
        
        # Crossover (random weighted average of parents)
        n_c = int(np.round((pop - n_e) * c_f))       # number of crossover children
        
        for j in range(0,n_c):
            i_pair = np.ceil(n_r * np.random.rand(2,1))       # generate random pair of parents
            weight = np.random.rand(8)                        # generate random weighting
            # Crossover parents by weighted blend to generate new child           
            x_new[(n_e + j),:] = weight * x_mate[int(i_pair[0,0])-1,:] + (1 - weight) * x_mate[int(i_pair[1,0])-1,:]
        
        # Mutation (gaussian noise added to parents)
        n_m = pop - n_e - n_c;       # number of mutation children
        
        for k in range(0,n_m):
            # Select random parent from mating pool and add white noise
            x_new[(n_e + n_c + k),:] = np.abs(x_mate[int(np.ceil(n_r * np.random.rand(1)) - 1),:] + sigma * np.random.randn(8))
        
        x = x_new
        
        # Check solution of current generation
        for i in range(0,pop):
            diff = np.subtract(pqt, calc_pqt(sf,x[i,:]))
            y = np.divide(diff, pqt)
            err[i] = np.dot(y, np.transpose(y))
        
            if err[i] < err_tol:
                z = x[i,:]
                conv = 1
                return z, gen, err[i], conv
        
        # If the last generation, then output best results
        if gen == n_gen:
            fitness = np.sort(err)
            index = np.argsort(err)
            z = x[index[0],:]
            conv = 0
            err = fitness[0]
            return z, gen, err, conv
