#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Moto: Induction motor parameter estimation tool

Hybrid Algorithms

Author: Julius Susanto
Last edited: August 2014
"""
import numpy as np
import globals
from descent import nr_solver, dnr_solver, lm_solver

"""
HY_SOLVER  - Hybrid algorithm solver for double cage model with core losses
             Solves for 6 circuit parameters [Xs Xm Rr1 Xr1 Rr2 Rc]
             Rs and Xr2 are computed by linear restrictions
             Includes change of variables
             Includes adaptive step size (as per Pedra 2008)
             Includes determinant check of jacobian matrix

Usage: hy_solver (p, desc, pop, n_r, n_e, c_f, n_gen, err_tol)

Where   p is a vector of motor performance parameters:
        p = [sf eff pf Tb Tlr Ilr]
          sf = full-load slip
          eff = full-load efficiency
          pf = full-load power factor
          T_b = breakdown torque (as # of FL torque)
          T_lr = locked rotor torque (as # of FL torque)
          I_lr = locked rotor current
        desc is the type of descent algorithm used - "NR", "LM", "DNR"
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
def hy_solver(self, desc, p, pop, n_r, n_e, c_f, n_gen, err_tol):
    
    # Initial settings
    gen = 1
    conv = 0  
    sigma = 0.01
    
    # Create initial population of Rs and Xr2 estimates
    RX = 0.15 * np.random.rand(pop,2)
    x = np.zeros((pop,8))
    iter = np.zeros(pop)
    err = np.zeros(pop)
    conv = np.zeros(pop)
    
    # Check solution of initial population
    for i in range(0,pop):
        self.statusBar().showMessage('Calculating generation %d, member %d...' % (gen, i+1))
        
        if desc == "NR":
            [x[i,:], iter[i], err[i], conv[i]] = nr_solver(p, 1, RX[i,0], RX[i,1], globals.algo_data["max_iter"], globals.algo_data["conv_err"])
        
        if desc == "LM":
            [x[i,:], iter[i], err[i], conv[i]] = lm_solver(p, 1, RX[i,0], RX[i,1], 1e-7, 5.0, globals.algo_data["max_iter"], globals.algo_data["conv_err"])
            
        if desc == "DNR":
            [x[i,:], iter[i], err[i], conv[i]] = dnr_solver(p, 1, RX[i,0], RX[i,1], 1e-7, globals.algo_data["max_iter"], globals.algo_data["conv_err"])
        
        if err[i] < err_tol:
            z = x[i,:]
            conv = 1
            return z, gen, err[i], conv
    
    # Run genetic algorithm
    for gen in range(2,n_gen+1):
        
        # Select for fitness
        fitness = np.sort(err)
        index = np.argsort(err)
        
        # Create next generation
        RX_new = np.zeros((pop,2))
        RX_mate = RX[index[0:n_r],:]         # select mating pool
        
        # Elite children (select best "n_e" children for next generation)
        RX_new[0:n_e,:] = RX[index[0:n_e],:]
        
        # Crossover (random weighted average of parents)
        n_c = int(np.round((pop - n_e) * c_f))       # number of crossover children
        
        for j in range(0,n_c):
            i_pair = np.ceil(n_r * np.random.rand(2,1))       # generate random pair of parents
            weight = np.random.rand(2)                        # generate random weighting
            # Crossover parents by weighted blend to generate new child           
            RX_new[(n_e + j),:] = weight * RX_mate[int(i_pair[0,0])-1,:] + (1 - weight) * RX_mate[int(i_pair[1,0])-1,:]
        
        # Mutation (gaussian noise added to parents)
        n_m = pop - n_e - n_c;       # number of mutation children
        
        for k in range(0,n_m):
            # Select random parent from mating pool and add white noise
            RX_new[(n_e + n_c + k),:] = np.abs(RX_mate[int(np.ceil(n_r * np.random.rand(1)) - 1),:] + sigma * np.random.randn(2))
        
        RX = RX_new
        
        # Check solution of current generation
        for i in range(0,pop):
            self.statusBar().showMessage('Calculating generation %d, member %d...' % (gen, i+1))
            
            if desc == "NR":
                [x[i,:], iter[i], err[i], conv[i]] = nr_solver(p, 1, RX[i,0], RX[i,1], globals.algo_data["max_iter"], globals.algo_data["conv_err"])
            
            if desc == "LM":
                [x[i,:], iter[i], err[i], conv[i]] = lm_solver(p, 1, RX[i,0], RX[i,1], 1e-7, 5.0, globals.algo_data["max_iter"], globals.algo_data["conv_err"])
                
            if desc == "DNR":
                [x[i,:], iter[i], err[i], conv[i]] = dnr_solver(p, 1, RX[i,0], RX[i,1], 1e-7, globals.algo_data["max_iter"], globals.algo_data["conv_err"])
                       
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
