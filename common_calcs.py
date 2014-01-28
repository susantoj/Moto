#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Moto: Induction motor parameter estimation tool

Common Calculations Module

Author: Julius Susanto
Last edited: January 2014
"""
import numpy as np

"""
GET_TORQUE - Calculate double cage motor torque and stator current (without core loss component)
 
Usage: get_torque (slip,type,x)

Where slip is the motor slip (pu)
       x is a vector of motor equivalent parameters:
           x = [Rs Xs Xm Rr1 Xr1 Rr2 Xr2]
            Rs = stator resistance
            Rs = stator reactance
            Xm = magnetising reactance
            Rr1 = rotor / inner cage resistance
            Xr1 = rotor / inner cage reactance
            Rr2 = outer cage resistance
            Xr2 = outer cage reactance       
             
Returns: motor torque (pu) as a real number and stator current (as a
complex number and without core loss component)
"""    
def get_torque(slip, x):
    
    # Calculate admittances
    Ys = 1 / np.complex(x[0], x[1])
    Ym = 1 / np.complex(0, x[2])
    Yr1 = 1 / np.complex(x[3] / slip, x[4])
    Yr2 = 1 / np.complex(x[5] / slip, x[6])
    
    # Calculate voltage and currents
    u1 = Ys / (Ys + Ym + Yr1 + Yr2)
    ir1 = np.abs (u1 * Yr1)
    ir2 = np.abs (u1 * Yr2)
    
    # Calculate torque and stator current
    torque = x[3] / slip * (ir1 ** 2) + x[5] / slip * (ir2 ** 2);
    ist = (1 - u1) * Ys
    
    return torque, ist

"""
CALC_PQT - Calculates motor mechanical power, reactive power, breakdown
torque and efficiency from equivalent circuit parameters (used for double
cage model with core losses)

Usage: calc_pqt (sf,x)

Where sf is the full load slip (pu)
       x is a 8 x 1 vector of motor equivalent parameters:
           x = [Rs Xs Xm Rr1 Xr1 Rr2 Xr2 Rc]
            x(0) = Rs = stator resistance
            x(1) = Xs = stator reactance
            x(2) = Xm = magnetising reactance
            x(3) = Rr1 = rotor / inner cage resistance
            x(4) = Xr1 = rotor / inner cage reactance
            x(5) = Rr2 = outer cage resistance
            x(6) = Xr2 = outer cage reactance
            x(7) = Rc = core resistance
              
Returns: y is a vector [Pm Q Tb I_nl]
"""
def calc_pqt(sf, x):

    x = np.abs(x)
    
    # Calculate full-load torque and current
    [T_fl, i_s] = get_torque(sf,x)
    
    # Calculate mechanical power (at FL)
    Pm = T_fl * (1 - sf)                               
    Sn = np.complex(1,0) * np.conj(i_s)
    
    # Calculate reactive power input (at FL)
    Q_fl = np.abs(np.imag(Sn)) 

    # Calculate core loss currents (at FL)
    i_c = 1 / np.complex (x[7],0)

    # Calculate total input current (at FL)    
    i_in = i_s + i_c

    # Calculate input power (at FL)
    p_in = np.real(np.complex(1,0) * np.conj(i_in))
    
    # Calculate efficiency (at FL)
    eff_fl = Pm / p_in                                 
    
    # Calculate breakdown torque with an interval search
    T_b = 0
    for n in range(1,101):
        i = float(n) / 100
        [T_i, I_i] = get_torque(i,x)                                
        if T_i > T_b:
            T_b = T_i           # Calculated breakdown torque

    [T_lr, i_lr] = get_torque(1,x);
    y = [Pm, Q_fl, T_b, T_lr, np.abs(i_lr + i_c), eff_fl]
    
    return y