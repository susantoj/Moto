#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Moto: Induction motor parameter estimation tool

Main window

Author: Julius Susanto
Last edited: January 2014
"""

import os, sys
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
import dateutil, pyparsing
import matplotlib.pyplot as plt
import globals
from saveload import *
from common_calcs import *
from descent import *
from genetic import *
from hybrid import *

class Window(QtGui.QMainWindow):
    
    def __init__(self):
        super(Window, self).__init__()
        
        globals.init()
        self.initUI()
        self.centre()
        
    def initUI(self):
        
        self.resize(800, 600)
        
        # Set background colour of main window to white
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Background,QtCore.Qt.white)
        self.setPalette(palette)
        
        self.setWindowTitle('SPE Moto | Induction Motor Parameter Estimation Tool')
        self.setWindowIcon(QtGui.QIcon('icons\motor.png'))    
              
        """
        Actions
        """
        exitAction = QtGui.QAction(QtGui.QIcon('icons\exit.png'), '&Exit', self)        
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.qApp.quit)
        
        loadAction = QtGui.QAction('&Open File...', self)
        loadAction.setStatusTip('Open file and load motor data')
        loadAction.triggered.connect(self.load_action)
        
        saveAction = QtGui.QAction('&Save As...', self)
        saveAction.setStatusTip('Save motor data')
        saveAction.triggered.connect(self.save_action)
        
        aboutAction = QtGui.QAction('&About Moto', self)
        aboutAction.setStatusTip('About Moto')
        aboutAction.triggered.connect(self.about_dialog)
        
        helpAction = QtGui.QAction('&User Manual', self)
        helpAction.setShortcut('F1')
        helpAction.setStatusTip('Moto user documentation')
        helpAction.triggered.connect(self.user_manual)
        
        """
        Menubar
        """
        menu_bar = self.menuBar() 
        fileMenu = menu_bar.addMenu('&File')
        fileMenu.addAction(loadAction)
        fileMenu.addAction(saveAction)
        fileMenu.addAction(exitAction)
        helpMenu = menu_bar.addMenu('&Help')
        helpMenu.addAction(helpAction)
        helpMenu.addSeparator()
        helpMenu.addAction(aboutAction)
        
        """
        Main Screen
        """
        
        heading_font = QtGui.QFont()
        heading_font.setPointSize(10)
        heading_font.setBold(True)
        
        ################
        # Motor details
        ################
        
        header1 = QtGui.QLabel('Motor Details')
        #header1.setMinimumWidth(50)
        header1.setMinimumHeight(30)
        header1.setFont(heading_font)
        
        label1 = QtGui.QLabel('Description')
        #label1.setMinimumWidth(50)
        
        self.le1 = QtGui.QLineEdit()
        #self.le1.setMinimumWidth(150)
        self.le1.setText(str(globals.motor_data["description"]))
               
        label2 = QtGui.QLabel('Synchronous speed')
        #label2.setMinimumWidth(50)
        
        self.le2 = QtGui.QLineEdit()
        #self.le2.setMinimumWidth(50)
        self.le2.setText(str(globals.motor_data["sync_speed"]))
        
        label2a = QtGui.QLabel('rpm')
        #label2a.setMinimumWidth(30)
 
        label3 = QtGui.QLabel('Rated speed')
        #label3.setMinimumWidth(50)
        
        self.le3 = QtGui.QLineEdit()
        #self.le3.setMinimumWidth(50)
        self.le3.setText(str(globals.motor_data["rated_speed"]))
        
        label3a = QtGui.QLabel('rpm')
        #label3a.setMinimumWidth(30)
           
        label4 = QtGui.QLabel('Rated power factor')
        #label4.setMinimumWidth(50)
        
        self.le4 = QtGui.QLineEdit()
        #self.le4.setMinimumWidth(50)
        self.le4.setText(str(globals.motor_data["rated_pf"]))
        
        label4a = QtGui.QLabel('pf')
        #label4a.setMinimumWidth(20)
        
        label5 = QtGui.QLabel('Rated efficiency')
        #label5.setMinimumWidth(50)
        
        self.le5 = QtGui.QLineEdit()
        #self.le5.setMinimumWidth(50)
        self.le5.setText(str(globals.motor_data["rated_eff"]))
        
        label5a = QtGui.QLabel('pu')
        #label5a.setMinimumWidth(20)

        label6 = QtGui.QLabel('Breakdown torque')
        #label6.setMinimumWidth(50)
        
        self.le6 = QtGui.QLineEdit()
        #self.le6.setMinimumWidth(50)
        self.le6.setText(str(globals.motor_data["T_b"]))
        
        label6a = QtGui.QLabel('T/Tn')
        #label6a.setMinimumWidth(40)
        
        label7 = QtGui.QLabel('Locked rotor torque')
        #label7.setMinimumWidth(50)
        
        self.le7 = QtGui.QLineEdit()
        #self.le7.setMinimumWidth(50)
        self.le7.setText(str(globals.motor_data["T_lr"]))
        
        label7a = QtGui.QLabel('T/Tn')
        #label7a.setMinimumWidth(40)
        
        label8 = QtGui.QLabel('Locked rotor current')
        #label8.setMinimumWidth(50)
        
        self.le8 = QtGui.QLineEdit()
        #self.le8.setMinimumWidth(50)
        self.le8.setText(str(globals.motor_data["I_lr"]))
        
        label8a = QtGui.QLabel('pu')
        #label8a.setMinimumWidth(40)
        
        ########
        # Model
        ########
        
        header2 = QtGui.QLabel('Motor Model')
        header2.setMinimumHeight(40)
        header2.setFont(heading_font)
        
        label_model = QtGui.QLabel('Model')
        #label_model.setMinimumWidth(150)
        
        self.combo_model = QtGui.QComboBox()
        # self.combo_model.addItem("Single cage")
        # self.combo_model.addItem("Single cage w/o core losses")
        self.combo_model.addItem("Double cage")
        
        img1 = QtGui.QLabel()
        img1.setPixmap(QtGui.QPixmap('images\dbl_cage.png'))
        
        #####################
        # Algorithm settings
        #####################
        
        header3 = QtGui.QLabel('Algorithm Settings')
        header3.setMinimumHeight(40)
        header3.setFont(heading_font)
        
        label9 = QtGui.QLabel('Maximum # iterations')
        
        self.le9 = QtGui.QLineEdit()
        self.le9.setText(str(globals.algo_data["max_iter"]))
        self.le9.setStatusTip('Maximum number of iterations allowed')
        
        label10 = QtGui.QLabel('Convergence criterion')
        
        self.le10 = QtGui.QLineEdit()
        self.le10.setText(str(globals.algo_data["conv_err"]))
        self.le10.setStatusTip('Squared error required to qualify for convergence')

        self.label11 = QtGui.QLabel('Linear constraint k_r')
        
        self.le11 = QtGui.QLineEdit()
        self.le11.setText(str(globals.algo_data["k_r"]))
        self.le11.setStatusTip('Linear constraint for Rs')

        self.label12 = QtGui.QLabel('Linear constraint k_x')
        
        self.le12 = QtGui.QLineEdit()
        self.le12.setText(str(globals.algo_data["k_x"]))
        self.le12.setStatusTip('Linear constraint for Xr2')
        
        # Genetic Algorithm Widgets
        ############################
        
        self.labeln_gen = QtGui.QLabel('Maximum # generations')
        self.labeln_gen.setVisible(0)
        self.labelpop = QtGui.QLabel('Members in population')
        self.labelpop.setVisible(0)
        self.labeln_r = QtGui.QLabel('Members in mating pool')
        self.labeln_r.setVisible(0)
        self.labeln_e = QtGui.QLabel('Elite children')
        self.labeln_e.setVisible(0)
        self.labelc_f = QtGui.QLabel('Crossover fraction')
        self.labelc_f.setVisible(0)
        
        self.len_gen = QtGui.QLineEdit()
        self.len_gen.setText(str(globals.algo_data["n_gen"]))
        self.len_gen.setStatusTip('Maximum number of generations allowed')
        self.len_gen.hide()
        
        self.lepop = QtGui.QLineEdit()
        self.lepop.setText(str(globals.algo_data["pop"]))
        self.lepop.setStatusTip('Number of members in each generation')
        self.lepop.hide()
        
        self.len_r = QtGui.QLineEdit()
        self.len_r.setText(str(globals.algo_data["n_r"]))
        self.len_r.setStatusTip('Number of members in a mating pool')
        self.len_r.hide()
        
        self.len_e = QtGui.QLineEdit()
        self.len_e.setText(str(globals.algo_data["n_e"]))
        self.len_e.setStatusTip('Number of elite children')
        self.len_e.hide()
        
        self.lec_f = QtGui.QLineEdit()
        self.lec_f.setText(str(globals.algo_data["c_f"]))
        self.lec_f.setStatusTip('Proportion of children spawned through crossover')
        self.lec_f.hide()
        
        
        label_algo = QtGui.QLabel('Algorithm')
        #label_algo.setMinimumWidth(150)
        
        self.combo_algo = QtGui.QComboBox()
        self.combo_algo.addItem("Newton-Raphson")
        self.combo_algo.addItem("Levenberg-Marquardt")
        self.combo_algo.addItem("Damped Newton-Raphson")
        self.combo_algo.addItem("Genetic Algorithm")
        self.combo_algo.addItem("Hybrid GA-NR")
        self.combo_algo.addItem("Hybrid GA-LM")
        self.combo_algo.addItem("Hybrid GA-DNR")
        
        calc_button = QtGui.QPushButton("Calculate")
        calc_button.setStatusTip('Estimate equivalent circuit parameters')
        
        self.plot_button = QtGui.QPushButton("Plot")
        self.plot_button.setDisabled(1)
        self.plot_button.setStatusTip('Plot torque-speed and current-speed curves')
        
        ####################
        # Algorithm results
        ####################
        
        header4 = QtGui.QLabel('Algorithm Results')
        #header4.setMinimumWidth(150)
        header4.setMinimumHeight(40)
        header4.setFont(heading_font)
        
        label13 = QtGui.QLabel('R_s')
        #label13.setFixedWidth(50)
        
        self.leRs = QtGui.QLineEdit()
        self.leRs.setStatusTip('Stator resistance (pu)')
        
        label14 = QtGui.QLabel('X_s')
        #label14.setMinimumWidth(150)
        
        self.leXs = QtGui.QLineEdit()
        self.leXs.setStatusTip('Stator reactance (pu)')
        
        label15 = QtGui.QLabel('X_m')
        #label15.setMinimumWidth(150)
        
        self.leXm = QtGui.QLineEdit()
        self.leXm.setStatusTip('Magnetising resistance (pu)')
        
        label16 = QtGui.QLabel('X_r1')
        #label16.setMinimumWidth(150)
        
        self.leXr1 = QtGui.QLineEdit()
        self.leXr1.setStatusTip('Inner cage rotor reactance (pu)')
        
        label17 = QtGui.QLabel('R_r1')
        #label17.setMinimumWidth(150)
        
        self.leRr1 = QtGui.QLineEdit()
        self.leRr1.setStatusTip('Inner cage rotor resistance (pu)')
        
        label18 = QtGui.QLabel('X_r2')
        #label18.setMinimumWidth(150)
        
        self.leXr2 = QtGui.QLineEdit()
        self.leXr2.setStatusTip('Outer cage rotor reactance (pu)')
        
        label19 = QtGui.QLabel('R_r2')
        #label19.setMinimumWidth(150)
        
        self.leRr2 = QtGui.QLineEdit()
        self.leRr2.setStatusTip('Outer cage rotor resistance (pu)')
        
        label20 = QtGui.QLabel('R_c')
        #label20.setMinimumWidth(150)
        
        self.leRc = QtGui.QLineEdit()
        self.leRc.setStatusTip('Core loss resistance (pu)')
        
        label21 = QtGui.QLabel('Converged?')
        #label21.setMinimumWidth(150)
        
        self.leConv = QtGui.QLineEdit()
        self.leConv.setStatusTip('Did algorithm converge?')
        
        label22 = QtGui.QLabel('Squared Error')
        #label22.setMinimumWidth(150)
        
        self.leErr = QtGui.QLineEdit()
        self.leErr.setStatusTip('Squared error of estimate')
        
        label23 = QtGui.QLabel('Iterations')
        #label23.setMinimumWidth(150)
        
        self.leIter = QtGui.QLineEdit()
        self.leIter.setStatusTip('Number of iterations / generations')
        
        ##############
        # Grid layout
        ##############
        
        grid = QtGui.QGridLayout()
        
        # Motor details
        i = 0
        grid.addWidget(header1, i, 0)
        grid.addWidget(label1, i+1, 0)
        grid.addWidget(self.le1, i+1, 1, 1, 5)
        grid.addWidget(label2, i+2, 0)
        grid.addWidget(self.le2, i+2, 1)
        grid.addWidget(label2a, i+2, 2)
        grid.addWidget(label3, i+3, 0)
        grid.addWidget(self.le3, i+3, 1)
        grid.addWidget(label3a, i+3, 2)
        grid.addWidget(label4, i+4, 0)
        grid.addWidget(self.le4, i+4, 1)
        grid.addWidget(label4a, i+4, 2)
        grid.addWidget(label5, i+5, 0)
        grid.addWidget(self.le5, i+5, 1)
        grid.addWidget(label5a, i+5, 2)
        grid.addWidget(label6, i+3, 4)
        grid.addWidget(self.le6, i+3, 5)
        grid.addWidget(label6a, i+3, 6)
        grid.addWidget(label7, i+4, 4)
        grid.addWidget(self.le7, i+4, 5)
        grid.addWidget(label7a, i+4, 6)
        grid.addWidget(label8, i+5, 4)
        grid.addWidget(self.le8, i+5, 5)
        grid.addWidget(label8a, i+5, 6)
        
        # Model
        i = 10
        grid.addWidget(header2, i, 0)
        grid.addWidget(label_model, i+1, 0)
        grid.addWidget(self.combo_model, i+1, 1)
        grid.addWidget(img1, i, 3, i-7, 6)
        
        # Algorithm settings
        i = 12
        grid.addWidget(header3, i, 0)
        grid.addWidget(label_algo, i+1, 0)
        grid.addWidget(self.combo_algo, i+1, 1)
        grid.addWidget(label9, i+2, 0)
        grid.addWidget(self.le9, i+2, 1)
        grid.addWidget(label10, i+3, 0)
        grid.addWidget(self.le10, i+3, 1)
        grid.addWidget(self.label11, i+2, 3)
        grid.addWidget(self.le11, i+2, 4)
        grid.addWidget(self.label12, i+3, 3)
        grid.addWidget(self.le12, i+3, 4)
        
        # Genetic algorithm parameters
        grid.addWidget(self.labeln_gen, i+2, 3)
        grid.addWidget(self.len_gen, i+2, 4)
        grid.addWidget(self.labelpop, i+3, 3)
        grid.addWidget(self.lepop, i+3, 4)
        grid.addWidget(self.labeln_r, i+4, 3)
        grid.addWidget(self.len_r, i+4, 4)
        grid.addWidget(self.labeln_e, i+2, 5)
        grid.addWidget(self.len_e, i+2, 6)
        grid.addWidget(self.labelc_f, i+3, 5)
        grid.addWidget(self.lec_f, i+3, 6)
        
        grid.addWidget(calc_button, 0, 5)
        grid.addWidget(self.plot_button, 0, 6)
        
        # Algorithm results
        i = 17
        grid.addWidget(header4, i, 0)
        grid.addWidget(label13, i+1, 0)
        grid.addWidget(self.leRs, i+1, 1)
        grid.addWidget(label14, i+2, 0)
        grid.addWidget(self.leXs, i+2, 1)
        grid.addWidget(label15, i+3, 0)
        grid.addWidget(self.leXm, i+3, 1)
        grid.addWidget(label20, i+4, 0)
        grid.addWidget(self.leRc, i+4, 1)
        grid.addWidget(label16, i+1, 3)
        grid.addWidget(self.leXr1, i+1, 4)
        grid.addWidget(label17, i+2, 3)
        grid.addWidget(self.leRr1, i+2, 4)
        grid.addWidget(label18, i+3, 3)
        grid.addWidget(self.leXr2, i+3, 4)
        grid.addWidget(label19, i+4, 3)
        grid.addWidget(self.leRr2, i+4, 4)
        grid.addWidget(label21, i+1, 5)
        grid.addWidget(self.leConv, i+1, 6)
        grid.addWidget(label22, i+2, 5)
        grid.addWidget(self.leErr, i+2, 6)
        grid.addWidget(label23, i+3, 5)
        grid.addWidget(self.leIter, i+3, 6)
        
        grid.setAlignment(Qt.AlignTop)      

        main_screen = QWidget()
        main_screen.setLayout(grid)
        main_screen.setStatusTip('Ready')
        
        self.setCentralWidget(main_screen)
        
        # Event handlers
        calc_button.clicked.connect(self.calculate)
        self.plot_button.clicked.connect(self.plot_curves)
        
        self.le1.editingFinished.connect(self.update_data)
        self.le2.editingFinished.connect(self.update_data)
        self.le3.editingFinished.connect(self.update_data)
        self.le4.editingFinished.connect(self.update_data)
        self.le5.editingFinished.connect(self.update_data)
        self.le6.editingFinished.connect(self.update_data)
        self.le7.editingFinished.connect(self.update_data)
        self.le8.editingFinished.connect(self.update_data)
        self.le9.editingFinished.connect(self.update_data)
        self.le10.editingFinished.connect(self.update_data)
        self.le11.editingFinished.connect(self.update_data)
        self.le12.editingFinished.connect(self.update_data)
        self.len_gen.editingFinished.connect(self.update_data)
        self.lepop.editingFinished.connect(self.update_data)
        self.len_r.editingFinished.connect(self.update_data)
        self.len_e.editingFinished.connect(self.update_data)
        self.lec_f.editingFinished.connect(self.update_data)
        
        ##########################
        #TO DO - connects for combo boxes - combo_model and combo_algo (what signal to use?)
        ##########################
        self.combo_algo.currentIndexChanged.connect(self.update_algo)
        
        self.statusBar().showMessage('Ready')
    
    # Calculate parameter estimates
    def calculate(self):
        self.statusBar().showMessage('Calculating...')
        
        sf = (globals.motor_data["sync_speed"] - globals.motor_data["rated_speed"]) / globals.motor_data["sync_speed"]
        p = [sf, globals.motor_data["rated_eff"], globals.motor_data["rated_pf"], globals.motor_data["T_b"], globals.motor_data["T_lr"], globals.motor_data["I_lr"] ]
        
        if self.combo_algo.currentText() == "Newton-Raphson":
            [z, iter, err, conv] = nr_solver(p, 0, globals.algo_data["k_x"], globals.algo_data["k_r"], globals.algo_data["max_iter"], globals.algo_data["conv_err"])           
        
        if self.combo_algo.currentText() == "Levenberg-Marquardt":
            [z, iter, err, conv] = lm_solver(p, 0, globals.algo_data["k_x"], globals.algo_data["k_r"], 1e-7, 5.0, globals.algo_data["max_iter"], globals.algo_data["conv_err"])
            
        if self.combo_algo.currentText() == "Damped Newton-Raphson":
            [z, iter, err, conv] = dnr_solver(p, 0, globals.algo_data["k_x"], globals.algo_data["k_r"], 1e-7, globals.algo_data["max_iter"], globals.algo_data["conv_err"])
            
        if self.combo_algo.currentText() == "Genetic Algorithm":
            [z, iter, err, conv] = ga_solver(self, p, globals.algo_data["pop"], globals.algo_data["n_r"], globals.algo_data["n_e"], globals.algo_data["c_f"], globals.algo_data["n_gen"], globals.algo_data["conv_err"])
            
        if self.combo_algo.currentText() == "Hybrid GA-NR":
            [z, iter, err, conv] = hy_solver(self, "NR", p, globals.algo_data["pop"], globals.algo_data["n_r"], globals.algo_data["n_e"], globals.algo_data["c_f"], globals.algo_data["n_gen"], globals.algo_data["conv_err"])
            
        if self.combo_algo.currentText() == "Hybrid GA-LM":
            [z, iter, err, conv] = hy_solver(self, "LM", p, globals.algo_data["pop"], globals.algo_data["n_r"], globals.algo_data["n_e"], globals.algo_data["c_f"], globals.algo_data["n_gen"], globals.algo_data["conv_err"])
            
        if self.combo_algo.currentText() == "Hybrid GA-DNR":
            [z, iter, err, conv] = hy_solver(self, "DNR", p, globals.algo_data["pop"], globals.algo_data["n_r"], globals.algo_data["n_e"], globals.algo_data["c_f"], globals.algo_data["n_gen"], globals.algo_data["conv_err"])
        
        self.leRs.setText(str(np.round(z[0],5)))
        self.leXs.setText(str(np.round(z[1],5)))
        self.leXm.setText(str(np.round(z[2],5)))
        self.leRr1.setText(str(np.round(z[3],5)))
        self.leXr1.setText(str(np.round(z[4],5)))
        self.leRr2.setText(str(np.round(z[5],5)))
        self.leXr2.setText(str(np.round(z[6],5)))
        self.leRc.setText(str(np.round(z[7],5)))
        
        if conv == 1:
            self.leConv.setText("Yes")
        else:
            QtGui.QMessageBox.warning(self, 'Warning', "Algorithm did not converge.", QtGui.QMessageBox.Ok)
            self.leConv.setText("No")
            
        self.leErr.setText(str(np.round(err,9)))
        self.leIter.setText(str(iter))
        
        # Only enable the plot button if the squared error is within the bounds of reason
        if err < 1:
            self.plot_button.setEnabled(1)
        else:
            self.plot_button.setDisabled(1)
        
        self.statusBar().showMessage('Ready')
        
    # Plot torque-speed and current-speed curves
    def plot_curves(self):
        sf = (globals.motor_data["sync_speed"] - globals.motor_data["rated_speed"]) / globals.motor_data["sync_speed"]
        x = [float(self.leRs.text()), float(self.leXs.text()) , float(self.leXm.text()), float(self.leRr1.text()), float(self.leXr1.text()), float(self.leRr2.text()), float(self.leXr2.text()), float(self.leRc.text())]
        
        # Rated per-unit torque
        T_rtd = globals.motor_data["rated_eff"] * globals.motor_data["rated_pf"] / (1 - sf)
        
        Tm = np.zeros(1001)
        Im = np.zeros(1001)
        speed = np.zeros(1001)
        speed[1000] = globals.motor_data["sync_speed"]
        for n in range(0,1000):
            speed[n] = float(n) / 1000 * globals.motor_data["sync_speed"]
            i = 1 - float(n) / 1000
            [Ti, Ii] = get_torque(i,x)
            
            Tm[n] = Ti / T_rtd      # Convert torque to T/Tn value
            Im[n] = np.abs(Ii)
        
        # Plot torque-speed and current-speed curves
        if plt.fignum_exists(1):
            # Do nothing
            QtGui.QMessageBox.warning(self, 'Warning', "A plot is already open. Please close to create a new plot.", QtGui.QMessageBox.Ok)
        else:
            plt.figure(1, facecolor='white')
            plt.subplot(211)
            plt.plot(speed, Tm)
            plt.xlim([0, globals.motor_data["sync_speed"]])
            plt.xlabel("Speed (rpm)")
            plt.ylabel("Torque (T/Tn)")
            plt.grid(color = '0.75', linestyle='--', linewidth=1)
            
            plt.subplot(212)
            plt.plot(speed, Im, 'r')
            plt.xlim([0, globals.motor_data["sync_speed"]])
            plt.xlabel("Speed (rpm)")
            plt.ylabel("Current (pu)")
            plt.grid(color = '0.75', linestyle='--', linewidth=1)
            
            plt.show()
    
    # Update global variables on change in data fields
    def update_data(self):
        globals.motor_data["description"] = str(self.le1.text())
        globals.motor_data["sync_speed"] = float(self.le2.text())
        globals.motor_data["rated_speed"] = float(self.le3.text())
        globals.motor_data["rated_pf"] = float(self.le4.text())
        globals.motor_data["rated_eff"] = float(self.le5.text())
        globals.motor_data["T_b"] = float(self.le6.text())
        globals.motor_data["T_lr" ] = float(self.le7.text())
        globals.motor_data["I_lr"] = float(self.le8.text())
        globals.algo_data["max_iter"] = int(self.le9.text())
        globals.algo_data["conv_err"] = float(self.le10.text())
        globals.algo_data["k_r"] = float(self.le11.text())
        globals.algo_data["k_x"] = float(self.le12.text())
        globals.algo_data["n_gen"] = int(self.len_gen.text())
        globals.algo_data["pop"] = int(self.lepop.text())
        globals.algo_data["n_r"] = int(self.len_r.text())
        globals.algo_data["n_e"] = int(self.len_e.text())
        globals.algo_data["c_f"] = float(self.lec_f.text())
    
    # Update data in the main window
    def update_window(self):
        self.le1.setText(str(globals.motor_data["description"]))
        self.le2.setText(str(globals.motor_data["sync_speed"]))
        self.le3.setText(str(globals.motor_data["rated_speed"]))
        self.le4.setText(str(globals.motor_data["rated_pf"]))
        self.le5.setText(str(globals.motor_data["rated_eff"]))
        self.le6.setText(str(globals.motor_data["T_b"]))
        self.le7.setText(str(globals.motor_data["T_lr"]))
        self.le8.setText(str(globals.motor_data["I_lr"]))
        
        self.le9.setText(str(globals.algo_data["max_iter"]))
        self.le10.setText(str(globals.algo_data["conv_err"]))
        self.le11.setText(str(globals.algo_data["k_r"]))
        self.le12.setText(str(globals.algo_data["k_x"]))
        self.len_gen.setText(str(globals.algo_data["n_gen"]))
        self.lepop.setText(str(globals.algo_data["pop"]))
        self.len_r.setText(str(globals.algo_data["n_r"]))
        self.len_e.setText(str(globals.algo_data["n_e"]))
        self.lec_f.setText(str(globals.algo_data["c_f"]))
    
    # Update the screen if the algorithm changes
    def update_algo(self):
        if (self.combo_algo.currentText() == "Genetic Algorithm") or (self.combo_algo.currentText() == "Hybrid GA-LM") or (self.combo_algo.currentText() == "Hybrid GA-NR") or (self.combo_algo.currentText() == "Hybrid GA-DNR"):
                self.label11.setVisible(0)
                self.le11.hide()
                self.label12.setVisible(0)
                self.le12.hide()
                
                self.labeln_gen.setVisible(1)
                self.labelpop.setVisible(1)
                self.labeln_r.setVisible(1)
                self.labeln_e.setVisible(1)
                self.labelc_f.setVisible(1)
                self.len_gen.show()
                self.lepop.show()
                self.len_r.show()
                self.len_e.show()
                self.lec_f.show()
        else:
                self.label11.setVisible(1)
                self.le11.show()
                self.label12.setVisible(1)
                self.le12.show()
                
                self.labeln_gen.setVisible(0)
                self.labelpop.setVisible(0)
                self.labeln_r.setVisible(0)
                self.labeln_e.setVisible(0)
                self.labelc_f.setVisible(0)
                self.len_gen.hide()
                self.lepop.hide()
                self.len_r.hide()
                self.len_e.hide()
                self.lec_f.hide()
                
    # Open file and load motor data
    def load_action(self):
        # Open file dialog box
        filename = QtGui.QFileDialog.getOpenFileName(self, "Open Moto File", "library/", "Moto files (*.mto)")
        
        if filename <> "":
            load_file(filename)
            self.update_window()
    
    # Save motor data to file
    def save_action(self):
        # Open save file dialog box
        filename = QtGui.QFileDialog.getSaveFileName(self, "Save Moto File", "library/", "Moto files (*.mto)")
        
        if filename <> "":
            save_file(filename)
    
    # Launch user manual
    def user_manual(self):
        os.system("start docs/moto_user_manual.pdf")
    
    # About dialog box
    def about_dialog(self):
        QtGui.QMessageBox.about(self, "About Moto",
                """<b>Moto</b> is a parameter estimation tool that can be used to determine the equivalent circuit parameters of induction machines. The tool is intended for use in dynamic time-domain simulations such as stability and motor starting studies.
                   <p>
                   Version: <b>v0.1 Beta<b><P>
                   <p>
                   Website: <a href="http://www.sigmapower.com.au/moto.html">www.sigmapower.com.au/moto.html</a>
                   <p> </p>
                   <p><img src="images/Sigma_Power.png"></p>
                   <p>&copy; 2014 Sigma Power Engineering Pty Ltd</p>
                   <p>All rights reserved.</p>
                   <p>
                   Redistribution and use in binary form is permitted provided that the following conditions are met:
                   <p>
                    1. Redistributions in binary form must reproduce the above copyright
                       notice, this list of conditions and the following disclaimer in the
                       documentation and/or other materials provided with the distribution.
                   <p>
                    2. All advertising materials mentioning features or use of this software
                       must display the following acknowledgement:
                       This product includes software developed by the Sigma Power Engineering Pty Ltd.
                   <p>
                    3. Neither the name of the Sigma Power Engineering Pty Ltd nor the
                       names of its contributors may be used to endorse or promote products
                       derived from this software without specific prior written permission.
                   <p>
                    THIS SOFTWARE IS PROVIDED BY SIGMA POWER ENGINEERING PTY LTD ''AS IS'' AND ANY
                    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
                    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
                    DISCLAIMED. IN NO EVENT SHALL SIGMA POWER ENGINEERING PTY LTD BE LIABLE FOR ANY
                    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
                    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
                    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
                    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
                    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
                    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                   
                   """)
    
    # Centre application window on screen
    def centre(self):
        qr = self.frameGeometry()
        qr.moveCenter(QtGui.QDesktopWidget().availableGeometry().center())
        
        self.move(qr.topLeft())

def main():
    
    app = QtGui.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()