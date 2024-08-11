# -*- coding: utf-8 -*- 
"""
Created on Tue Sep 13 13:39:23 2022

Author of base version: s183839
"""
import numpy as np
from scipy.integrate import odeint
import math
from sympy import Symbol
import pandas as pd
import pandas as pd
import numpy as np

class A_oryzae:
    def __init__(self, init_config, bio_config=None, fault_events=None):
        """
        Init function to load all parameters, inputs, and initial conditions.
        It should be coupled to PDI to recieve the current process inputs.
        An option could be to load all parameters from excel.
        An option could be to couple the class to a parameter estimation function to update the parameters.
           
        Input:
            Config file with adjustable starting parameters
            Optional config file for fault event
            
        Output:
            Dataframe with state variables and tracked variables at each time step     
              
        """
        self.state_variables = ['X', 'S', 'E', 'DO', 'V']
        self.default_paramters_output = ['DO_saturation', 'mu', 'EDCF', 'mu_app', 'kLa', 'F', 'F_set', 'P', 'N', 'air_L', 'air_L_set', 'OUR', 'OTR', 'weight','P_air', 'rho', 'rho_f', 'V_water', 'c_f']

        ### Design factors ###
        self.c_f_init = init_config['c_f'] # [g/L] Substrate feed concentration FIXED
        self.N_init = init_config['N'] # [rps = 1/s] Agitation FIXED
        self.p_HS = init_config['p_HS'] # Headspace pressure [bar] FIXED
        self.V0 = init_config['V0'] # [L] Initial volume [L] FIXED

        # Generate default bio_config is not provided
        if bio_config is None:
            bio_config = {
                "X0": 5.0,
                "C": 28.46,
                "Y_SX": 0.3,
                "Y_SE": 0.1,
                "Y_SO": 0.6,
                "Y_SC": 0.9,
                "Y_XS_true": 1.81,
                "Y_XO_true": 0.0328,
                "m_s": 0.013,
                "m_o": 0.0003
            }

        ### Optional fault event
        if fault_events is not None:
            self.fault_events = fault_events
        else:
            self.fault_events = None

        # Dict to track current state of faults (1 = True, 0 = False)
        self.current_faults= {
        'OUR_OOC': 0,
        'airflow_OOC': 0,
        'blocked_spargers': 0,
        'steam_in_feed': 0,
        'defect_steambarrier': 0
        }

        # Check for offgas calibration
        self.OUR_offset = self.check_offgas_calibration()

        # Initialize water added by steam (e.g. defect steam barrier and steam in feed)
        self.V_water = 0 # [L]

        ### Initialize changeable parameters ###
        self.N = self.N_init # [rps = 1/s] Agitation
        
        ### Inputs ###
        self.Q_air_start = 300 # [NL/min] start aeration
        self.Q_air_end = 300 # [NL/min] final aeration
        #self.N = 250/60 #  [rps = 1/s] agitation
        #self.c_f = 500 # [g/L] substrate feed concentration
        self.rho_f_init = 1100 # [g/L] feed density
        self.rho_f = self.rho_f_init # initialize feed density
        self.F_bias = 0.3 # [L/h] feed bias
        self.P_total_s = 3 # [kW/m^3] Total power input (changes?)
        self.P = 1280 # [W] Agitator power input (changes?)
        #self.p_HS = 1.01325 # [bar] headspace pressure 
        self.T_in=18 # [celcius] Inlet air temperature
        self.air_ramp_dur = 12.5 # [h] Duration of aeration ramp
        self.u_max_ramp = 0.00818181 # [L/(h^2)] ramp for maximum flow initially
        self.u_min_ramp = -0.004545 # [L/(h^2)] ramp for minimum flow initially
        self.u_ramp_bias = 0.2 # [L/h] flow controller bias during initial limitation
        self.u_max_f = 2 # [L/h] max flow after initial limitation
        self.u_min_f = 0.1 # [L/h] min flow after initial limitation
        self.DO_ramp_t = 180 # [h] flow rate ramp duration
        self.T = 28 # Degrees celcius
        
        ### Initial conditions (after batch phase) ###
        self.X0 = bio_config['X0'] # [g/L] Initial biomass concentration +- 10 %
        self.S0 = 0.05
        self.E0 = 0.05
        self.DO0 = 0.0004858076608546041 # less than do start initially
        #self.V0 = 300 # [L]
        self.Z0 = np.array([self.X0,self.S0, self.E0, self.DO0, self.V0])
        
        
        ### Impeller and reactor properties ###
        self.k_s = 11 # [-]  Metzner constant
        self.D = 0.298 # [m] Impeller diameter (B2-30 impeller)
        self.t_c = 0.5 # [s] Mean circulation time
        self.rad = 0.68/2 # [m] Radius of fermenter
        self.A = math.pi * (self.rad)**2 # [m^2] Cylindrical area of fermenter
        self.imp_splitheight = 0.332 # [m] Height of volume associated with bottom impeller
        self.imp_bWD = 0.1505/0.298 # [-] Bottom impeller width/diameter
        self.imp_bNV = 4   # [-] ?
        self.imp_tWD = 0.204/0.298 # [-] Top impeller width/diameter
        self.imp_tNV = 4 # [-] ?
        self.Po = 5 # Unaerated impeller power number for rushton disc turbine
        self.PgPo = 0.5 # Relative power draw upon aeration for rushton disc turbine
        self.imp_n = 2 # number of impellers
        
        ### kLa ###
        self.C = bio_config['C'] # [-] # Accuracy: +- 15 % (normal dist)
        self.a = 0.4544 # [-]
        self.b = 0.0538 # [-]
        self.c = -0.4516 # [-]
        
        ### Flow/Viscosity behavior ###
        self.C_1 = 0.0053 # [-]
        self.alpha = 1.56 # [-]
        self.beta = 0.060 # [-]
        self.C_2 = 2.69 # [-]
        self.delta = -0.568 # [-] 
        
        ### Bio-kinetic model ### # +- 10 % for each parameter (normal  dist)
        self.Y_SX = bio_config['Y_SX'] # [gDW/gS]
        self.Y_SE = bio_config['Y_SE'] # [gE/gS]
        self.Y_SO = bio_config['Y_SO'] # [gDO/gs]
        self.Y_SC = bio_config['Y_SC'] # [gCO2/gS]
        self.Y_XS_true = bio_config['Y_XS_true'] # [gS/gDW]
        self.Y_XO_true = bio_config['Y_XO_true'] # [moleO2/gDW]
        self.m_s = bio_config['m_s'] # [gS/(gDW*h)]
        self.m_o = bio_config['m_o'] # [moleO2/(gDW*h)]
        
        # Other
        self.M_mmO2 = 32 # [g/mole] Molar mass of oxygen
        self.M_mmCO2 = 44 # [g/mole] Molar mass of CO2
        self.p = 1.01325 # [bar] Atmospheric pressure
        self.DO_star = 0.00057317 # [Moles O2/L] Oxygen saturation (Calculate based on temp and pressure?)
        self.T_out=27 # [celcius] Temperature of off-gas
        self.rt=0.90 # [%/100] humidity
        self.rho_dry = 0.001292496 # [kg/NL] Density of dry air
        self.R_air = 2.870028305 # [L Bar/ K kg] Gas constant-air
        self.R_water = 4.614024417 # [L Bar/ K kg] Gas constant-water
        self.R = 0.08134472 # [L Bar/ K mol] Gas constant
        self.rho_broth = 1050 # [g/L] Density of broth
        self.g = 9.80665 # [m/s^2] Gravitational constant
        self.rho_water = 996.31 # [g/L] Density of water at 28 degrees celcius
        
        # Control parameters
        self.K_p = -40 # Proportional gain
        self.K_i = -400 # Interal gain
        self.t_sample = 0.1 # Sampling rate
        self.t_end = 230 # Simulation end time
        self.DOsp1 = 0.60*self.DO_star # Initial DO set point value
        self.DOsp2 = 0.40*self.DO_star # Final DO set point value
        
        
        # Max feed ramp
        self.max_time_vec = [0,14,24,28]
        self.max_val_vec = [0.2,0.9,1.5,2.0]
        self.min_time_vec = [0]
        self.min_val_vec = [0.1]

        #self.max_val_vec = #self.DO_df['vaerdi'][self.DO_df['funktion_kode'].str.contains('Højdos MAX out')]
        #self.min_time_vec = self.DO_df['tekst'][self.DO_df['funktion_kode'].str.contains('Højdos MIN out')]
        #self.min_time_vec = [int(i) for i in self.min_time_vec]
        #self.min_val_vec = self.DO_df['vaerdi'][self.DO_df['funktion_kode'].str.contains('Højdos MIN out')]
        self.max_i = 0
        self.min_i = 0
        
     
        
    def balance(self, state, r_Z, Z, V):
        """
        Calculates the reactor balance for a given state variable
        
        Input:
            state: String
                String specifying the state
            r_Z: Float
                Current float value of the rate of change for the specified state
            Z: Float
                Current float value of the specified state
            V: Float
                Current submerged volume of the reactor   
        
        """
        IN = 0
        if state == 'S':
            IN = self.c_f*self.F
        elif state == 'DO':
            IN = self.OTR_log*V 
    
        return IN/V + r_Z - Z/V*self.dVdt        
        
        
    def bio_kinetics(self):
        """
        Sets up the rate vector and the stoichiometric matrix with symbols
        
        Input:
            None
            
        Output:
            r: Numpy array
                Array that is the symbolic rate of change vector
            S_m: Numpy array 
                Array that is the symbolic stoichiometric matrix
        
        """
        Y_SX = Symbol('Y_SX')
        Y_SE = Symbol('Y_SE')
        Y_SC = Symbol('Y_SC')
        Y_SO = Symbol('Y_SO')
        Y_XS_true = Symbol('Y_XS_true')
        Y_XO_true = Symbol('Y_XO_true')
        m_s = Symbol('m_s')
        m_o = Symbol('m_o')
        mu = Symbol('mu')
        X = Symbol('X')

        S_m = np.array([[1, Y_XS_true, Y_SE*(1/Y_SX), -Y_XO_true],
                        [0,    -1    ,      0       ,       0   ],
                        [0,     0    ,      0       ,      -1   ]])
            
        rho_r = np.array([mu*X, m_s*X, m_o*X])
        r = np.matmul(np.transpose(S_m), rho_r)
        
        return r, S_m
        
    

    
    def model(self,Z,t,u):
        """
        Contains physical equations and computes the system of ODEs.
        
        Input:
            Z: Numpy array
                Array with the current values of the state variables
            t: Numpy array
                Array indicating the times the ODE solver should provide state values
            u: Float
                Float value of the PI controller to update substrate flow rate
        
        Output:
            dZdt_f: Numpy array
                Array with the current rate of change in the state variables
        
        """
        ### Volume balance###
        ## Evaporation ## 
        p_vap = 10**(5.1962-(1730.63/(self.T_in+233.5)))*self.rt # [bar] Vapor pressure
        rho_moist = (self.p_HS+self.p-p_vap)/(273.15+self.T_in)/self.R_air + self.p/(273.15+self.T_in)/self.R_water # [kg/L moist air]
        air = self.rho_dry* 60 / rho_moist # [-]
        n_in = p_vap/(self.R*(273.15+self.T_in)) # [mol/L]
        p_vap_out = 10**(5.1962-(1730.63/(self.T_out+233.5))) # [bar]
        n_out = p_vap_out/self.R/(273+self.T_out)*(self.p_HS+self.p-p_vap)/(self.p_HS+self.p-p_vap_out)*(273+self.T_out)/(273+self.T_in) # [mol/L]
        #self.F_evap = (n_out-n_in)*air*18.020*self.Q_air_end; # [g/h] ?????? units?
        self.F_evap = (n_out-n_in)*air*18.020*self.air_NL_corrected; # [g/h] evaporation calculation with corrected airflow

        # Feed flow
        self.F_set = u # pure feed stream without steam [L/h]
        self.F = u + self.feed_steam_in # [L/h]

        ## Density and feed concentration adjustment due to possible steam intake
        self.rho = (self.V_water*self.rho_water + (Z[4]-self.V_water)*self.rho_broth)/Z[4] # [g/L]
        self.rho_f = ((self.F-self.feed_steam_in) * self.rho_f_init + self.feed_steam_in * self.rho_water)/(self.F) # [g/L]	
        self.c_f = ((self.F-self.feed_steam_in) * self.c_f_init)/(self.F) # [g/L]

        # Weight balance
        self.weight = self.rho*Z[4]/1000 # [kg]
        
        # Change in volume
        self.dVdt = (self.rho_f*self.F + self.rho_water*self.barrier_steam_in - self.F_evap + self.Y_SO*self.F*self.c_f - self.Y_SC*self.F*self.c_f)/self.rho
        
        ### Bio-kinetic model ###
        # Specific growth rate
        self.mu = ((self.c_f*self.F/Z[4]-(Z[1]/Z[4]*self.dVdt))/Z[0] - self.m_s)/self.Y_XS_true # [1/h] (Eq. 9 Albaek 2011)
        # Stoichiometric matrix
        S_m = np.array([[1, -self.Y_XS_true, self.Y_SE*(1/self.Y_SX), -self.Y_XO_true],
                        [0,      -1        ,              0         ,          0     ],
                        [0,       0        ,              0         ,         -1     ]])
        rho_r = np.array([self.mu*Z[0], self.m_s*Z[0], self.m_o*Z[0]])
        r = np.matmul(np.transpose(S_m), rho_r)
        
        
        ### Physical equations to calculate kLa ###
        v_g = self.air_L/(1000*self.A*60) # [m/s] Superficial gas velocity
        
           ## EDCF ##
        bottom_po = -0.009312*self.N+6.12308 # Empirical correlation?******
        top_po = 5.5 # Known?****
        # Agitator power input (Albaek 2012, Eq. 9)
        self.P = (self.imp_n*self.Po*self.rho*(self.N**3)*(self.D**5)*self.PgPo) #[W]
        # Power dissipated by aeration (Roels and Heijnen, 1980)
        vg_standard = self.air_L/(1000*self.A*60)/(self.T+273.15)/(293.15)*self.p/(self.p_HS*+self.p) # [m/s] Superficial gas velocity
        h_ungassed = Z[4]/(1000*np.pi*self.rad**2) # Height of ungassed liquid in column [m]
        p_outlet = ((self.rho*self.g*h_ungassed)/(10**5))+self.p_HS # [bar] Absolute pressure at vessel outlet)
        self.P_air = ((vg_standard*self.R*(self.T+273.15)*Z[4]/1000)/(22.4*h_ungassed)*np.log(1+((self.rho*self.g*h_ungassed)/p_outlet))) # Z: ungassed height of liquid in column, p_0 = absolute pressure at vessel outlet
        #self.P_air = self.P_air/1000 # [kW] to [W]
        # Total energy dissipated in broth
        self.P_broth = self.P + self.P_air

        if self.N != 0: # Avoid division by zero
                # Bottom impeller #
            # Volume:
            bottom_V = self.imp_splitheight*self.A # [m^3]
            # Flow number:
            bottom_fl = 0.76*bottom_po 
            EDCF_b = (self.P/1000*bottom_po/(bottom_po+top_po))/(self.imp_bWD*math.pi/24*self.imp_bNV*bottom_V/(bottom_fl*self.N))
            
                # Top impeller #
            # Volume:
            imp_tV = Z[4]/1000-bottom_V # [m^3]
            # Flow number
            top_fl = 0.76*top_po**(1/3)
            EDCF_t = (self.P/1000*top_po/(bottom_po+top_po))/(math.pi/24*self.imp_tWD*self.imp_tNV*imp_tV/(top_fl*self.N))
            
            self.EDCF = (EDCF_b + EDCF_t)/2 # [W/(m^3*s) = Pa/s^2]
        else:
            self.EDCF = 0
         
        self.n = self.C_2*Z[0]**self.delta # [-] (Eq. 3b Albaek 2011)
        self.K = self.C_1*Z[0]**(self.alpha)*self.EDCF**(self.beta) # [Pa.s^n] (Eq. 3a Albaek 2011)
        gamma_eff = self.k_s*self.N # [1/s]
        tau = self.K*gamma_eff**self.n # [Pa]
        self.mu_app = tau/gamma_eff # [Pa.s]
        self.kLa = self.C*(self.P_total_s)**self.a*v_g**self.b*self.mu_app**self.c # [1/h] (Eq. 15 Albaek 2011)
        self.OTR = self.kLa*(self.DO_star-Z[3]) # mol/(L*h) (change to logarithmic driving force?)
        
        ### Logarithmic driving force ### (include varying HS pressure)
        # Henry's constant
        He = math.e**(14.835722+5837.177*(self.T+273.15)**(-1)-1085201*(self.T+273.15)**(-2))/100000*0.018015 # [bar*kg/molO2]
        # Oxygen solubility
        self.DO_star2 = (self.p+self.p_HS)*0.2094*self.rho/1000/He # [molO2/L]
        # Oxygen saturation in percent
        self.DO_saturation = Z[3]/self.DO_star2*100 # [%]
        # pressure at electrode
        liq_height = Z[4]/(1000*self.A) # [m]
        p_liq = (liq_height-0.3)*(self.A)*1.05/(self.A)/(10197.2/1000) # [bar] ???? 10197.2 mmwater/bar ???
        p_electrode = p_liq+self.p+self.p_HS # [bar]
        p_O2 = Z[3]/((1.01325+1.3)*0.2094/He)*364/750.1 # [bar] ????
  
        #self.OUR = -r[3]*Z[4]/ # [mol/h]
        self.OUR = -r[3] # [mol/L/h]
        # OUR3 = (self.mu*self.Y_XO_true+self.m_o)*Z[0]*Z[4]
        OUR2 = 8.31*(28+273.15)*self.OUR/(self.p_HS+1.01325)/100000 # [m^3 O2/h]
        O2 = (0.2094-(((self.air_L/1000*60*0.2094)-OUR2)*1000/self.air_L/60))*100 # [-] Difference in in- and outlet oxygen ratio???
        O2_out = ((self.air_L/1000*60*0.2094)-OUR2)*1000/self.air_L/60 # [-] Ratio of oxygen in outlet air
        logdrivingforce=((p_electrode*0.2094-p_O2)/He-(O2_out*(1.01325+self.p_HS)-p_O2)/He)/(math.log10((p_electrode*0.2094-p_O2)/He)-math.log10((O2_out*(1.01325+self.p_HS)-p_O2)/He))
        self.OTR_log = self.kLa*logdrivingforce
        
                
        # Mass balances of states previously:
        # dXdt = self.fed_batch_balance('X',r[0], Z[0], Z[4])
        # dSdt = self.fed_batch_balance('S',r[1], Z[1], Z[4])
        # dEdt = self.fed_batch_balance('E',r[2], Z[2], Z[4])
        # dDOdt = self.fed_batch_balance('DO',r[3], Z[3], Z[4])
        
        # Instead, a loop is set up to find all state balances except volume
        state = ('X', 'S', 'E', 'DO')
        dZdt = np.zeros(len(state))
        for i in range(len(state)):
            dZdt[i] = self.balance(state[i],r[i],Z[i],Z[4])
        
        # Return the state array as a list and concatenate with the change in volume
        dZdt_f = list(dZdt) + [self.dVdt]
        
        return dZdt_f
    
       

    def control_solve(self, parameters_output=None, track_state_var=True, target_column=False):
        """
        Contains PI controller and the aeration recipe. Calls ODE solver and returns the current state values.
        Should be made compatible with PDI.
        
        Input:
            None
        
        Output:
            Z_vec: Numpy array
                Array with the state variable values at the corresponding times
            Other: Numpy array
                Arrays with the values of the given parameter at the corresponding times
        
        """
        if parameters_output is not None:
            self.tracked_variables = parameters_output
        else:
            self.tracked_variables = self.default_paramters_output
        
        # Solve with discrete PI control, updating control parameter every self.t_sample hours
        t_num = np.linspace(0, self.t_sample, 10)
        t_meas = np.linspace(0,self.t_end,int(self.t_end/self.t_sample))
        
        # Aerartion rate vector
        air_NL = np.zeros(len(t_meas))

        # Define output dataframe from state variables and tracked variables
        #output_df = pd.DataFrame(0, index=range(0,len(t_meas)), columns=['t'] + self.state_variables + self.tracked_variables, dtype=np.float64)
        output_df = pd.DataFrame(0, index=range(0,len(t_meas)), columns=['t'], dtype=np.float64)
        
        u_vec = np.zeros(len(t_meas)) # controller output
        e = np.zeros(len(t_meas)) # error
        ie = np.zeros(len(t_meas)) # integral of the error
        P = np.zeros(len(t_meas)) # proportional term
        I = np.zeros(len(t_meas)) # integral term
        
        Z_vec = np.zeros((len(t_meas),5)) # state variables: X, S, E, DO, V
        
        
        for i in range(len(t_meas)-1):
            ### Check for possible event and apply faults ###
            N_fault = self.check_fault(t_meas[i], 'N')
            if N_fault is not None:
                self.N = N_fault
            else:
                self.N = self.N_init

            ### Check steam barriers and track possible steam intake
            self.check_steam(t_meas[i])
            self.V_water = self.V_water + (self.feed_steam_in+self.barrier_steam_in)*self.t_sample # [L]

            ### PI control of DO through feed rate ###
            DO_p_set = self.DOsp1 - t_meas[i]*(self.DOsp1-self.DOsp2)/self.DO_ramp_t  
            # Error
            e[i] = DO_p_set - self.Z0[3]
            # Sum of errors
            if i >= 1:
                ie[i] = ie[i-1] + e[i]*self.t_sample
            
            P[i] = self.K_p*e[i]
            I[i] = self.K_i*ie[i]
            u_vec[i]=P[i]+I[i]
            u=u_vec[i] + self.F_bias
            
            
            ### PDI feed ###
            if self.max_i == len(self.max_time_vec)-1:
                self.u_max = self.max_val_vec[self.max_i]
                
            elif t_meas[i] >= self.max_time_vec[self.max_i] and t_meas[i] < self.max_time_vec[self.max_i+1] :
                self.u_max = self.max_val_vec[self.max_i]
            
            else:
                self.max_i += 1
                
            self.u_min = 0.1
            
            # if i == 130:
            #     print(self.u_max)
            #     print(t_meas[i])
            
            # if i == 130:
            #     print(self.u_max)
            #     print(t_meas[i])
            # if i == 139:
            #     print(self.u_max)
            #     print(t_meas[i])
            # if i == 141:
            #     print(self.u_max)
            #     print(t_meas[i])
            # if i == 239:
            #     print(self.u_max)
            #     print(t_meas[i])
            # if i == 241:
            #     print(self.u_max)
            #     print(t_meas[i])
            # if i == 279:
            #     print(self.u_max)
            #     print(t_meas[i])
            # if i == 281:
            #     print(self.u_max)
            #     print(t_meas[i])
            
            
            # Limitation on feeding initially
            #if t_meas[i]<=22:
             #   self.u_max = self.u_max_ramp*t_meas[i]+self.u_ramp_bias
              #  self.u_min = self.u_min_ramp*t_meas[i]+self.u_ramp_bias
            #else:
             #   self.u_max = self.u_max_f
              #  self.u_min = self.u_min_f
 
            # Ensure max and min flow rate is not exceeded
            if u>self.u_max:
                u=self.u_max
                ie[i] = ie[i] - e[i]*self.t_sample # anti-reset windup
            if u+self.F_bias < self.u_min:
                u =self.u_min
                ie[i] = ie[i] - e[i]*self.t_sample # anti-reset windup

            feed_fault = self.check_fault(t_meas[i], 'F')
            if feed_fault is not None:
                u = feed_fault
            
            ### Aeration ###
            # The batch phase lasts until 23 hours, from where the simulation runs. Here begins the aeration ramp
            # and it lasts 12.5 hours.
            if t_meas[i] < self.air_ramp_dur:
                air_NL[i] = self.Q_air_start - (self.Q_air_start-self.Q_air_end)/self.air_ramp_dur*t_meas[i]
            else:
                air_NL[i] = self.Q_air_end
            
            self.air_L = air_NL[i]*(self.T+273.15)/(293.15)*self.p/(self.p_HS*+self.p) # [L/min]

            # Save aeration setpoint for the model output
            self.air_L_set = self.air_L

            # Check for blocked spargers and out-of-calibration airflow meter
            spargers_fault = self.check_aeration(t_meas[i])
            if spargers_fault is not None:
                self.air_L = self.air_L*(1+(spargers_fault/100))

            # Recalculate air_NL for evaporation correction
            self.air_NL_corrected = self.air_L/(self.T+273.15)/(293.15)*self.p/(self.p_HS*+self.p) # [NL/min]

            ### Solve ODE ###
            Z = odeint(self.model, self.Z0, t_num, rtol = 1e-7, mxstep= 500000, args=(u,))

            ### Save initial parameters for plotting ###
            if i == 0:                
                output_df.at[i, 't'] = t_meas[i] # Save time
                if track_state_var == True:
                    for variable in self.state_variables:
                        output_df.at[i, variable] = self.Z0[self.state_variables.index(variable)] # Save state variables

                for variable in self.tracked_variables:
                    output_df.at[i, variable] = getattr(self, variable) # Save tracked variables

                # check if offgas sensor is calibrated. If not, apply offset to OUR in output df
                if self.OUR_offset is not None:
                    output_df.at[i, 'OUR'] = self.OUR*(1+(self.OUR_offset/100))
                
                # Track target columns
                if target_column:
                    fault_counter = 0
                    for column, state in self.current_faults.items():
                        output_df.at[i, column] = state
                        fault_counter += state
                    if fault_counter == 0:
                        output_df.at[i, 'no_fault'] = 1
                    else:
                        output_df.at[i, 'no_fault'] = 0
            
            ### Update states and parameters###
            self.Z0 = Z[-1,:]

            Z_vec[i+1,:] = self.Z0
            output_df.at[i+1, 't'] = t_meas[i+1]
            if track_state_var == True:
                for variable in self.state_variables:
                    output_df.at[i+1, variable] = self.Z0[self.state_variables.index(variable)]

            for variable in self.tracked_variables:
                output_df.at[i+1, variable] = getattr(self, variable)

            # check if offgas sensor is calibrated. If not, apply offset to OUR in output df
            if self.OUR_offset is not None:
                output_df.at[i+1, 'OUR'] = self.OUR*(1+(self.OUR_offset/100))

            # Track target columns
            if target_column:
                fault_counter = 0
                for column, state in self.current_faults.items():
                    output_df.at[i+1, column] = state
                    fault_counter += state
                if fault_counter == 0:
                    output_df.at[i+1, 'no_fault'] = 1
                else:
                    output_df.at[i+1, 'no_fault'] = 0
            
        return output_df
    def control_solve_save(self, ):
        output = self.control_solve()
        output.to_csv('../data/output.csv')
        return output

    def check_fault(self, t, variable_name):
        """
        Check if there is a fault occurring for the given variable at the current time.
        
        Args:
        t (float): Current time.
        variable_name (str): The name of the fault variable.
        
        Returns:
        float or None: The fault value if a fault is occurring, otherwise None.
        """
        # Iterate through each fault event
        if self.fault_events is not None:
            for event in self.fault_events:
                # Check if the event's parameter matches the given variable and if the current
                # time falls within the event's start and end times.
                if (event['event_type'] == variable_name and
                        event['t_start'] <= t <= event['t_end']):
                    # A matching fault event is found, return the fault value
                    return event['fault_value']
        
        # No matching fault event found, return None
        return None

    def check_offgas_calibration(self):
        """
        Check if there is a fault in offgas calibration.
        
        Args:
        None.
        
        Returns:
        float or None: The fault value if a fault is occurring, otherwise None.
        """
        # Iterate through each fault event
        self.current_faults['OUR_OOC'] = 0
        if self.fault_events is not None:
            for event in self.fault_events:
                # Check if OUR OOC event is occurring
                if (event['event_type'] == "OUR_OOC"):
                    # A matching fault event is found, return the fault value
                    self.current_faults['OUR_OOC'] = 1
                    return event['offset']
        
        # No matching fault event found, return None
        return None
    
    def check_aeration(self, t):
        """
        Check for blocked spargers and out-of-calibration airflow meter.
        
        Args:
        None.
        
        Returns:
        float or None: The fault value if a fault is occurring, otherwise None.
        """
        # Iterate through each fault event
        self.current_faults['airflow_OOC'] = 0
        self.current_faults['blocked_spargers'] = 0
        if self.fault_events is not None:
            offset = 0
            for event in self.fault_events:
                if (event['event_type'] == "airflow_OOC"):
                    self.current_faults['airflow_OOC'] = 1
                    offset = event['offset']
                #elif (event['event_type'] == "airflow_OOC" and event['constant'] == False and event['t_start'] <= t):
                    
                # Check if spargers are blocked is occurring
                if (event['event_type'] == "blocked_spargers" and
                    event['t_start'] <= t):
                        self.current_faults['blocked_spargers'] = 1
                        t_slope = t - event['t_start']
                        if t_slope < event['duration']:
                            offset = offset + event['offset']*(t_slope/event['duration'])
                        else:
                            offset = offset + event['offset']
            return offset
        
        # No matching fault event found, return None
        return None
    
    def check_steam(self, t):
        """
        Check if spargers are blocked.
        
        Args:
        None.
        
        Returns:
        float or None: The fault value if a fault is occurring, otherwise None.
        """
        # Initialize steam flow values
        self.feed_steam_in = 0
        self.barrier_steam_in = 0
        # Iterate through each fault event
        if self.fault_events is not None:
            for event in self.fault_events:
                if (event['event_type'] == "steam_in_feed" and
                    event['t_start'] <= t <= event['t_start'] + event['duration']):
                    self.feed_steam_in = event['steamflow'] # [L/h]
                if (event['event_type'] == "defect_steambarrier" and
                    event['t_start'] <= t <= event['t_start'] + event['duration']):
                    self.barrier_steam_in = event['steamflow'] # [L/h]

        if self.feed_steam_in > 0:
            self.current_faults['steam_in_feed'] = 1
        else:
            self.current_faults['steam_in_feed'] = 0

        if self.barrier_steam_in > 0:
            self.current_faults['defect_steambarrier'] = 1
        else:
            self.current_faults['defect_steambarrier'] = 0