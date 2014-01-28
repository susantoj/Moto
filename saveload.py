#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Moto: Induction motor parameter estimation tool

Save/Load Module

Author: Julius Susanto
Last edited: January 2014
"""
import os, sys
import globals

# Load motor data from save file and put data into globals object
def load_file(filename):
    i = 1
    for line in open(filename):
        [key, item] = line.split(";")
        
        if i <= 8:
            if key == "description":
                globals.motor_data[key] = str(item)
            else:
                globals.motor_data[key] = float(item)
        else:
            if (key == "max_iter") or (key == "n_gen") or (key == "pop") or (key == "n_r") or (key == "n_e"):
                globals.algo_data[key] = int(item)
            else:
                globals.algo_data[key] = float(item)
        
        i = i + 1

def save_file(filename):
    f = open(filename, "w")
    
    f.write("description;%s\n" % globals.motor_data["description"])
    f.write("sync_speed;%f\n" % globals.motor_data["sync_speed"])
    f.write("rated_speed;%f\n" % globals.motor_data["rated_speed"])
    f.write("rated_pf;%f\n" % globals.motor_data["rated_pf"])
    f.write("rated_eff;%f\n" % globals.motor_data["rated_eff"])
    f.write("T_b;%f\n" % globals.motor_data["T_b"])
    f.write("T_lr;%f\n" % globals.motor_data["T_lr"])
    f.write("I_lr;%f\n" % globals.motor_data["I_lr"])
    
    f.write("max_iter;%d\n" % globals.algo_data["max_iter"])
    f.write("k_r;%f\n" % globals.algo_data["k_r"])
    f.write("k_x;%f\n" % globals.algo_data["k_x"])
    f.write("conv_err;%f\n" % globals.algo_data["conv_err"])
    f.write("n_gen;%d\n" % globals.algo_data["n_gen"])
    f.write("pop;%d\n" % globals.algo_data["pop"])
    f.write("n_r;%d\n" % globals.algo_data["n_r"])
    f.write("n_e;%d\n" % globals.algo_data["n_e"])
    f.write("c_f;%f\n" % globals.algo_data["c_f"])
    
    f.close()