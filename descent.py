#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Moto: Induction motor parameter estimation tool

Descent Algorithms

Author: Julius Susanto
Last edited: January 2014
"""
import numpy as np
from common_calcs import *

"""
NR_SOLVER  - Newton-Rhapson solver for double cage model with core losses
             Solves for 6 circuit parameters [Xs Xm Rr1 Xr1 Rr2 Rc]
             Includes change of variables
             Includes adaptive step size (as per Pedra 2008)
             Includes determinant check of jacobian matrix

Usage: nr_solver (p, kx, kr, max_iter, err_tol)

Where   p is a vector of motor performance parameters:
        p = [sf eff pf Tb Tlr Ilr]
          sf = full-load slip
          eff = full-load efficiency
          pf = full-load power factor
          T_b = breakdown torque (as # of FL torque)
          T_lr = locked rotor torque (as # of FL torque)
          I_lr = locked rotor current
        mode = 0: normal, 1: fixed Rs and Xr2
        kx and kr are linear restrictions in normal mode
                  and fixed Xr2 and Kr in mode 1
        max_iter is the maximum number of iterations  
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
def nr_solver(p, mode, kx, kr, max_iter, err_tol):
    
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

    # Set initial conditions
    z = np.zeros(8)
    z[2] = 1 / Q_fl            #Xm
    z[1] = 0.05 * z[2]         #Xs
    z[3] = 1 / Pm_fl * sf      #Rr1
    z[4] = 1.2 * z[1]          #Xr1
    z[5] = 5 * z[3]            #Rr2
    z[7] = 12
    
    if mode == 0:
        z[0] = kr * z[3]           #Rs
        z[6] = kx * z[1]           #Xr2
    else:
        z[0] = kr
        z[6] = kx
    
    # Change of variables to constrained parameters (with initial values)
    x = np.zeros(6)
    x[0] = z[3]
    x[1] = z[5] - z[3]
    x[2] = z[2]
    x[3] = z[1]
    x[4] = z[4] - z[6]
    x[5] = z[7]
    
    # Formulate solution
    pqt = [Pm_fl, Q_fl, T_b, T_lr, i_lr, eff]

    # Set up NR algorithm parameters
    h = 0.00001
    n = 0
    hn = 1
    hn_min = 0.0000001
    err = 1.0
    iter = 0
    conv = 0
    
    # Run NR algorithm
    while (err > err_tol) and (iter < max_iter):
        
        # Evaluate objective function for current iteration
        diff = np.subtract(pqt, calc_pqt(sf,z))
        y = np.divide(diff, pqt)
        err0 = np.dot(y, np.transpose(y))
        
        # Construct Jacobian matrix
        j = np.zeros((6,6))
        for i in range(1,7):
            x[i-1] = x[i-1] + h
            
            # Change of variables back to equivalent circuit parameters
            z[1] = x[3]
            z[2] = x[2]
            z[3] = x[0]
            
            if mode == 0:
                z[4] = kx * x[3] + x[4]
            else:
                z[4] = z[6] + x[4]
            
            z[5] = x[0] + x[1]
            z[7] = x[5]
            
            if mode == 0:
                z[0] = kr * z[3]
                z[6] = kx * z[1]
            
            diff = np.subtract(pqt, calc_pqt(sf,z))
            j[:,i-1] = (np.divide(diff, pqt) - y) / h
            x[i-1] = x[i-1] - h
        
        # Check if jacobian matrix is singular and exit function if so
        if (np.linalg.det(j) == 0):
            print "Jacobian matrix is singular"
            break
        
        x_reset = x
        y_reset = y
        iter0 = iter
        
        # Inner loop (descent direction check and step size adjustment)
        while (iter == iter0):
            # Calculate next iteration and update x
            jmat = np.matrix(j)
            delta_x = np.dot(jmat.getI(), np.transpose(y)).A[0]
            x = np.abs(np.subtract(x, hn * delta_x))
            
            # Change of variables back to equivalent circuit parameters
            z[1] = x[3]
            z[2] = x[2]
            z[3] = x[0]
            
            if mode == 0:
                z[4] = kx * x[3] + x[4]
            else:
                z[4] = z[6] + x[4]
            
            z[5] = x[0] + x[1]
            z[7] = x[5]
            
            if mode == 0:
                z[0] = kr * z[3]
                z[6] = kx * z[1]
            
            # Calculate squared error terms
            diff = np.subtract(pqt, calc_pqt(sf,z))
            y = np.divide(diff, pqt)
            err = np.dot(y, np.transpose(y))
            
            # Descent direction check and step size adjustment
            if (np.abs(err) >= np.abs(err0)):
                n = n + 1
                hn = 2 ** (-n)
                x = x_reset
                y = y_reset
            else:
                n = 0
                iter = iter + 1
            
            # If descent direction isn't minimising, then there is no convergence
            if (hn < hn_min):
                break 

    if err < err_tol:
        conv = 1
    
    return z, iter, err, conv

"""
LM_SOLVER  - Levenberg-Marquadt solver for double cage model with core losses
             Solves for 6 circuit parameters [Xs Xm Rr1 Xr1 Rr2 Rc]
             Includes change of variables
             Includes adaptive step size (as per Pedra 2008)
             Includes determinant check of jacobian matrix
             Basic error adjustment of damping parameter lambda

Usage: lm_solver (p, mode, kx, kr, lambda_0, lambda_max, max_iter, err_tol)

Where   p is a vector of motor performance parameters:
        p = [sf eff pf Tb Tlr Ilr]
          sf = full-load slip
          eff = full-load efficiency
          pf = full-load power factor
          T_b = breakdown torque (as # of FL torque)
          T_lr = locked rotor torque (as # of FL torque)
          I_lr = locked rotor current
        mode = 0: normal, 1: fixed Rs and Xr2
        kx and kr are linear restrictions in normal mode
                  and fixed Xr2 and Kr in mode 1
        lambda_0 is initial damping parameter
        lambda_max is maximum damping parameter
        max_iter is the maximum number of iterations
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
def lm_solver(p, mode, kx, kr, lambda_0, lambda_max, max_iter, err_tol):
    
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

    # Set initial conditions
    z = np.zeros(8)
    z[2] = 1 / Q_fl            #Xm
    z[1] = 0.05 * z[2]         #Xs
    z[3] = 1 / Pm_fl * sf      #Rr1
    z[4] = 1.2 * z[1]          #Xr1
    z[5] = 5 * z[3]            #Rr2
    z[7] = 12
    
    if mode == 0:
        z[0] = kr * z[3]           #Rs
        z[6] = kx * z[1]           #Xr2
    else:
        z[0] = kr
        z[6] = kx
    
    # Change of variables to constrained parameters (with initial values)
    x = np.zeros(6)
    x[0] = z[3]
    x[1] = z[5] - z[3]
    x[2] = z[2]
    x[3] = z[1]
    x[4] = z[4] - z[6]
    x[5] = z[7]
    
    # Formulate solution
    pqt = [Pm_fl, Q_fl, T_b, T_lr, i_lr, eff]

    # Set up LM algorithm parameters
    h = 0.00001
    lambda_i = lambda_0
    err = 1.0
    iter = 0
    conv = 0
    beta = 3
    gamma = 3
    
    # Run LM algorithm
    while (err > err_tol) and (iter < max_iter):
        
        # Evaluate objective function for current iteration
        diff = np.subtract(pqt, calc_pqt(sf,z))
        y = np.divide(diff, pqt)
        err0 = np.dot(y, np.transpose(y))
        
        # Construct Jacobian matrix
        j = np.zeros((6,6))
        for i in range(1,7):
            x[i-1] = x[i-1] + h
            
            # Change of variables back to equivalent circuit parameters
            z[1] = x[3]
            z[2] = x[2]
            z[3] = x[0]
            
            if mode == 0:
                z[4] = kx * x[3] + x[4]
            else:
                z[4] = z[6] + x[4]
            
            z[5] = x[0] + x[1]
            z[7] = x[5]
            
            if mode == 0:
                z[0] = kr * z[3]
                z[6] = kx * z[1]
            
            diff = np.subtract(pqt, calc_pqt(sf,z))
            j[:,i-1] = (np.divide(diff, pqt) - y) / h
            x[i-1] = x[i-1] - h
        
        # Check if jacobian matrix is singular and exit function if so
        if (np.linalg.det(j) == 0):
            print "Jacobian matrix is singular"
            break
        
        x_reset = x
        y_reset = y
        iter0 = iter
        
        # Inner loop (descent direction check and step size adjustment)
        while (iter == iter0):
            # Calculate next iteration and update x
            # (Matlab: delta_x = inv(j'*j + lambda_i.*diag(diag(j'*j)))*j'*y')
            
            jblock = np.dot(np.transpose(j), j)
            j1 = jblock + lambda_i * np.diag(np.diag(jblock))
            j2 = np.matrix(j1)
            j3 = np.dot(j2.getI(), np.transpose(j))
            
            delta_x = np.dot(j3, np.transpose(y)).A[0]
            x = np.abs(np.subtract(x, delta_x))
            
            # Change of variables back to equivalent circuit parameters
            z[1] = x[3]
            z[2] = x[2]
            z[3] = x[0]
            
            if mode == 0:
                z[4] = kx * x[3] + x[4]
            else:
                z[4] = z[6] + x[4]
            
            z[5] = x[0] + x[1]
            z[7] = x[5]
            
            if mode == 0:
                z[0] = kr * z[3]
                z[6] = kx * z[1]
            
            # Calculate squared error terms
            diff = np.subtract(pqt, calc_pqt(sf,z))
            y = np.divide(diff, pqt)
            err = np.dot(y, np.transpose(y))
            
            ####################
            # TO DO
            #if (isnan(err)):
            #    err = 6;
            
            # Error adjustment of lambda
            if (np.abs(err) >= np.abs(err0)) and iter > 0:
                lambda_i = lambda_i * beta;
                x = x_reset
                y = y_reset
            else:
                lambda_i = lambda_i / gamma
                iter = iter + 1
            
            # If descent direction isn't minimising, then there is no convergence
            if (lambda_i > lambda_max):
                break 

    if err < err_tol:
        conv = 1
    
    return z, iter, err, conv

"""
DNR_SOLVER - Damped Newton-Rhapson solver for double cage model with core losses
             Solves for 6 circuit parameters [Xs Xm Rr1 Xr1 Rr2 Rc]
             Includes change of variables
             Includes adaptive step size (as per Pedra 2008)
             Includes determinant check of jacobian matrix

Usage: dnr_solver (p, mode, kx, kr, lambda_i, max_iter, err_tol)

Where   p is a vector of motor performance parameters:
        p = [sf eff pf Tb Tlr Ilr]
          sf = full-load slip
          eff = full-load efficiency
          pf = full-load power factor
          T_b = breakdown torque (as # of FL torque)
          T_lr = locked rotor torque (as # of FL torque)
          I_lr = locked rotor current
        mode = 0: normal, 1: fixed Rs and Xr2
        kx and kr are linear restrictions in normal mode
                  and fixed Xr2 and Kr in mode 1
        lambda_i is the initial damping parameter
        max_iter is the maximum number of iterations  
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
def dnr_solver(p, mode, kx, kr, lambda_i, max_iter, err_tol):
    
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

    # Set initial conditions
    z = np.zeros(8)
    z[2] = 1 / Q_fl            #Xm
    z[1] = 0.05 * z[2]         #Xs
    z[3] = 1 / Pm_fl * sf      #Rr1
    z[4] = 1.2 * z[1]          #Xr1
    z[5] = 5 * z[3]            #Rr2
    z[7] = 12
    
    if mode == 0:
        z[0] = kr * z[3]           #Rs
        z[6] = kx * z[1]           #Xr2
    else:
        z[0] = kr
        z[6] = kx
    
    # Change of variables to constrained parameters (with initial values)
    x = np.zeros(6)
    x[0] = z[3]
    x[1] = z[5] - z[3]
    x[2] = z[2]
    x[3] = z[1]
    x[4] = z[4] - z[6]
    x[5] = z[7]
    
    # Formulate solution
    pqt = [Pm_fl, Q_fl, T_b, T_lr, i_lr, eff]

    # Set up DNR algorithm parameters
    h = 0.00001
    n = 0
    hn = 1
    hn_min = 0.0000001
    err = 1.0
    iter = 0
    conv = 0
    gamma = 3
    beta = 3
    
    # Run DNR algorithm
    while (err > err_tol) and (iter < max_iter):
        
        # Evaluate objective function for current iteration
        diff = np.subtract(pqt, calc_pqt(sf,z))
        y = np.divide(diff, pqt)
        err0 = np.dot(y, np.transpose(y))
        
        # Construct Jacobian matrix
        j = np.zeros((6,6))
        for i in range(1,7):
            x[i-1] = x[i-1] + h
            
            # Change of variables back to equivalent circuit parameters
            z[1] = x[3]
            z[2] = x[2]
            z[3] = x[0]
            
            if mode == 0:
                z[4] = kx * x[3] + x[4]
            else:
                z[4] = z[6] + x[4]
            
            z[5] = x[0] + x[1]
            z[7] = x[5]
            
            if mode == 0:
                z[0] = kr * z[3]
                z[6] = kx * z[1]
            
            diff = np.subtract(pqt, calc_pqt(sf,z))
            j[:,i-1] = (np.divide(diff, pqt) - y) / h
            x[i-1] = x[i-1] - h
        
        # Check if jacobian matrix is singular and exit function if so
        if (np.linalg.det(j) == 0):
            print "Jacobian matrix is singular"
            break
        
        x_reset = x
        y_reset = y
        iter0 = iter
        
        # Inner loop (descent direction check and step size adjustment)
        while (iter == iter0):
            # Calculate next iteration and update x
            jmat = np.matrix(np.subtract(j, lambda_i * np.identity(6)))
            delta_x = np.dot(jmat.getI(), np.transpose(y)).A[0]
            x = np.abs(np.subtract(x, hn * delta_x))
            
            # Change of variables back to equivalent circuit parameters
            z[1] = x[3]
            z[2] = x[2]
            z[3] = x[0]
            
            if mode == 0:
                z[4] = kx * x[3] + x[4]
            else:
                z[4] = z[6] + x[4]
            
            z[5] = x[0] + x[1]
            z[7] = x[5]
            
            if mode == 0:
                z[0] = kr * z[3]
                z[6] = kx * z[1]
            
            # Calculate squared error terms
            diff = np.subtract(pqt, calc_pqt(sf,z))
            y = np.divide(diff, pqt)
            err = np.dot(y, np.transpose(y))
            
            # Descent direction check and step size adjustment
            if (np.abs(err) >= np.abs(err0)):
                n = n + 1
                hn = 2 ** (-n)
                lambda_i = lambda_i * beta
                x = x_reset
                y = y_reset
            else:
                n = 0
                lambda_i = lambda_i / gamma
                iter = iter + 1
            
            # If descent direction isn't minimising, then there is no convergence
            if (hn < hn_min):
                return z, iter, err, conv

    if err < err_tol:
        conv = 1
    
    return z, iter, err, conv