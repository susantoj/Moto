#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Moto: Induction motor parameter estimation tool

Global Objects and Variables

Author: Julius Susanto
Last edited: January 2014
"""

def init():
    global motor_data
    global algo_data
    
    motor_data = {
        "description"       : "Toshiba 6.6kV 350kW",
        "sync_speed"        : 1500.0,     
        "rated_speed"       : 1481.0,
        "rated_pf"          : 0.87,     
        "rated_eff"         : 0.91,
        "T_b"               : 3.2,     
        "T_lr"              : 2.4,     
        "I_lr"              : 6.5     
        }
        
    algo_data = {
        "max_iter"          : 30,     
        "k_r"               : 1.0,
        "k_x"               : 0.5,     
        "conv_err"          : 1e-5,
        "n_gen"             : 30,
        "pop"               : 20,
        "n_r"               : 15,
        "n_e"               : 2,
        "c_f"               : 0.8
        }
    
    