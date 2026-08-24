import MonteCarlo as mc

# Black-Scholes model
BS_model = mc.Black_Scholes(r = 0.05, sigma = 0.2)

Heston_model = mc.Heston(r = 0.05, sigma = 0.2, kappa = 2, theta = 0.04)
# Heston_model = mc.Heston(r = 0.05, sigma = 0.05, kappa = 0.5, theta = 0.9)

# Approximating schemes
EM_scheme = mc.Euler_Maruyama_scheme(BS_model)

M_scheme = mc.Milstein_scheme(BS_model)

Milstein_Heston = mc.log_Heston_Milstein(Heston_model)
    

# Options with T = 1 and K = 105 (K = 101 for the Digital option)
European_call_option = mc.European_call(T = 1, K = 105)
    
Asian_call_option = mc.Asian_call(T = 1, K = 105)

Lookback_call_option = mc.Lookback_call(T = 1, K = 105)

Digital_call_option = mc.Digital_call(T = 1, K = 101)


# Algorithms
Standard_MC = mc.Single_level_MC()

Multilevel_MC = mc.MLMC()


# Run the standard Single-level Monte Carlo algorithm
# and store the output in corresponding objects.
# Parameters: S_0 = 100, M = 10_000, N = 1000.
European_call_st_run = Standard_MC.run(M_scheme, European_call_option,
                                       S_0 = 100, M = 10_000, N = 1000, bridge = False)

Asian_call_st_run = Standard_MC.run(M_scheme, Asian_call_option,
                                    S_0 = 100, M = 10_000, N = 1000, bridge = True)

Lookback_call_st_run = Standard_MC.run(M_scheme, Lookback_call_option,
                                       S_0 = 100, M = 10_000, N = 1000, bridge = True)

Digital_call_st_run = Standard_MC.run(M_scheme, Digital_call_option,
                                      S_0 = 100, M = 10_000, N = 1000, bridge = True)

European_call_heston_st_run = Standard_MC.run_heston(Milstein_Heston, European_call_option, 
                                                     S_0 = 1, V_0 = 1, M = 10_000, N = 1000)

Asian_call_heston_st_run = Standard_MC.run_heston(Milstein_Heston, Asian_call_option, 
                                                     S_0 = 1, V_0 = 1, M = 10_000, N = 1000)


# Run the MLMC algorithm and store the output in corresponding objects.
# Parameters: S_0 = 100, M_in = 1000, N = 2, eps is option-specific.
# Warning: if the accuracy is chosen to be not enough small, the algorithm
# may perform only one iteration, providing too few levels to fit a linear model
# which causes an error.
European_call_MLMC_run = Multilevel_MC.run(M_scheme, European_call_option,
                                           S_0 = 100, M_in = 100, eps = 0.01,
                                           N = 2, bridge = False)

Asian_call_MLMC_run = Multilevel_MC.run(M_scheme, Asian_call_option,
                                        S_0 = 100, M_in = 100, eps = 0.001,
                                        N = 2, bridge = True)

Lookback_call_MLMC_run = Multilevel_MC.run(M_scheme, Lookback_call_option,
                                           S_0 = 100, M_in = 100, eps = 0.01,
                                           N = 2, bridge = True)

Digital_call_MLMC_run = Multilevel_MC.run(M_scheme, Digital_call_option,
                                          S_0 = 100, M_in = 100, eps = 0.00005,
                                          N = 2, bridge = True)

European_call_MLMC_heston_antithetic_run = Multilevel_MC.run_antithetic(Milstein_Heston, European_call_option,
                                                             S_0 = 1, V_0 = 1, M_in = 100, eps = 0.0003,
                                                             N = 2, antithetic = True)

European_call_MLMC_heston_run = Multilevel_MC.run_antithetic(Milstein_Heston, European_call_option,
                                                             S_0 = 1, V_0 = 1, M_in = 100, eps = 0.0003,
                                                             N = 2, antithetic = False)

Asian_call_MLMC_heston_antithetic_run = Multilevel_MC.run_antithetic(Milstein_Heston, Asian_call_option,
                                                             S_0 = 1, V_0 = 1, M_in = 100, eps = 0.001,
                                                             N = 2, antithetic = True)

Asian_call_MLMC_heston_run = Multilevel_MC.run_antithetic(Milstein_Heston, Asian_call_option,
                                                             S_0 = 1, V_0 = 1, M_in = 100, eps = 0.001,
                                                             N = 2, antithetic = False)

# Run MLMC_eps_sweep for a set of accuracies.
# Parameters: S_0 = 100, M_in = 1000, N = 2.
accuracy = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
European_call_MLMC_eps_sweep = Multilevel_MC.run_eps_sweep(M_scheme, European_call_option,
                                                           S_0 = 100, M_in = 100, eps_set = accuracy,
                                                           N = 2, bridge = False)

accuracy = [0.3, 0.1, 0.05, 0.01, 0.005, 0.002]
Asian_call_MLMC_eps_sweep = Multilevel_MC.run_eps_sweep(M_scheme, Asian_call_option,
                                                        S_0 = 100, M_in = 100, eps_set = accuracy,
                                                        N = 2, bridge = True)

accuracy = [0.3, 0.2, 0.1, 0.05, 0.03, 0.01]
Lookback_call_MLMC_eps_sweep = Multilevel_MC.run_eps_sweep(M_scheme, Lookback_call_option,
                                                           S_0 = 100, M_in = 100, eps_set = accuracy,
                                                           N = 2, bridge = True)

accuracy = [0.0025, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
Digital_call_MLMC_eps_sweep = Multilevel_MC.run_eps_sweep(M_scheme, Digital_call_option,
                                                          S_0 = 100, M_in = 100, eps_set = accuracy,
                                                          N = 2, bridge = True)

accuracy = [0.01, 0.005, 0.001, 0.0005, 0.0003]
European_call_MLMC_heston_eps_antithetic_sweep = Multilevel_MC.run_antithetic_eps_sweep(Milstein_Heston, European_call_option,
                                                                             S_0 = 1, V_0 = 1, M_in = 100, eps_set = accuracy,
                                                                             N = 2, antithetic = True)

European_call_MLMC_heston_eps_sweep = Multilevel_MC.run_antithetic_eps_sweep(Milstein_Heston, European_call_option,
                                                                             S_0 = 1, V_0 = 1, M_in = 100, eps_set = accuracy,
                                                                             N = 2, antithetic = False)

accuracy = [0.3, 0.1, 0.05, 0.01, 0.005, 0.001]
Asian_call_MLMC_heston_eps_antithetic_sweep = Multilevel_MC.run_antithetic_eps_sweep(Milstein_Heston, Asian_call_option,
                                                                             S_0 = 1, V_0 = 1, M_in = 100, eps_set = accuracy,
                                                                             N = 2, antithetic = True)

Asian_call_MLMC_heston_eps_sweep = Multilevel_MC.run_antithetic_eps_sweep(Milstein_Heston, Asian_call_option,
                                                                             S_0 = 1, V_0 = 1, M_in = 100, eps_set = accuracy,
                                                                             N = 2, antithetic = False)


# Run run_levels sweep for the standard Single-level Monte Carlo for a set of levels.
# Parameter L is chosen as large as the maximum level we may need for the Analysis.
European_call_st_sweep = Standard_MC.run_levels_sweep(M_scheme, European_call_option,
                                                      S_0 = 100, M = 10_000,
                                                      L = max(European_call_MLMC_run.L, *European_call_MLMC_eps_sweep.L),
                                                      N = 2, bridge = False)

Asian_call_st_sweep = Standard_MC.run_levels_sweep(M_scheme, Asian_call_option,
                                                   S_0 = 100, M = 10_000,
                                                   L = max(Asian_call_MLMC_run.L, *Asian_call_MLMC_eps_sweep.L),
                                                   N = 2, bridge = True)

Lookback_call_st_sweep = Standard_MC.run_levels_sweep(M_scheme, Lookback_call_option,
                                                      S_0 = 100, M = 10_000,
                                                      L = max(Lookback_call_MLMC_run.L, *Lookback_call_MLMC_eps_sweep.L),
                                                      N = 2, bridge = True)

Digital_call_st_sweep = Standard_MC.run_levels_sweep(M_scheme, Digital_call_option,
                                                     S_0 = 100, M = 10_000,
                                                     L = max(Digital_call_MLMC_run.L, *Digital_call_MLMC_eps_sweep.L),
                                                     N = 2, bridge = True)

European_call_heston_st_sweep = Standard_MC.run_heston_levels_sweep(Milstein_Heston, European_call_option,
                                                                    S_0 = 1, V_0 = 1, M = 10_000, L = max(European_call_MLMC_heston_antithetic_run.L, *European_call_MLMC_heston_eps_antithetic_sweep.L))

Asian_call_heston_st_sweep = Standard_MC.run_heston_levels_sweep(Milstein_Heston, Asian_call_option,
                                                                    S_0 = 100, V_0 = 0.04, M = 10_000, L = max(Asian_call_MLMC_heston_run.L, *Asian_call_MLMC_heston_eps_sweep.L))

# Run the Analysis
Analysis_MLMC = mc.Analysis()

Analysis_MLMC.comprehensive_plot(European_call_MLMC_run, European_call_MLMC_eps_sweep, European_call_st_sweep)

Analysis_MLMC.comprehensive_plot(Asian_call_MLMC_run, Asian_call_MLMC_eps_sweep, Asian_call_st_sweep)

Analysis_MLMC.comprehensive_plot(Lookback_call_MLMC_run, Lookback_call_MLMC_eps_sweep, Lookback_call_st_sweep)

Analysis_MLMC.comprehensive_plot(Digital_call_MLMC_run, Digital_call_MLMC_eps_sweep, Digital_call_st_sweep)

Analysis_MLMC.comprehensive_plot(European_call_MLMC_heston_antithetic_run, European_call_MLMC_heston_eps_antithetic_sweep, European_call_heston_st_sweep)

Analysis_MLMC.comprehensive_plot(Asian_call_MLMC_heston_run, Asian_call_MLMC_heston_eps_sweep, Asian_call_heston_st_sweep)



#########################################################

import numpy as np
import matplotlib.pyplot as plt


class Analysis_antithetic:

    def log_variance_plot(self, result_MLMC, result_MLMC_antithetic, result_sl_MC_sweep = None, ax = None):
        
        # Define semi-log axes
        l = np.arange(result_MLMC.L + 1)
        
        var_l = result_MLMC.var
        var_l_ant = result_MLMC_antithetic.var
        
        mask = var_l > 0
        mask_ant = var_l_ant > 0
                
        log_var = np.full(len(var_l), np.nan, dtype = float)
        log_var_ant = np.full(len(var_l_ant), np.nan, dtype = float)
        
        log_var[mask] = np.log(var_l[mask]) / np.log(result_MLMC.N)
        log_var_ant[mask_ant] = np.log(var_l_ant[mask_ant]) / np.log(result_MLMC_antithetic.N)
                
        if mask.sum() >= 3:
            # Fit a linear model. Take the last 3 point to capture the true
            # asymptotics and ignore the pre-asymptotics.
            slope, intercept = np.polyfit(l[mask][-3:], log_var[mask][-3:], deg = 1)
            slope_rounded = round(slope * 2) / 2
            
            # Reference line
            var_ref = np.linspace(1, result_MLMC.L, 50)
            ref_slope = (slope_rounded * var_ref +
                         (log_var[mask][-1] - slope_rounded * l[mask][-1] - 0.7))
            
        if mask_ant.sum() >= 3:
            # Fit a linear model. Take the last 3 point to capture the true
            # asymptotics and ignore the pre-asymptotics.
            slope_ant, intercept_ant = np.polyfit(l[mask_ant][-3:], log_var_ant[mask_ant][-3:], deg = 1)
            slope_rounded_ant = round(slope_ant * 2) / 2
            
            # Reference line
            var_ref_ant = np.linspace(1, result_MLMC_antithetic.L, 50)
            ref_slope_ant = (slope_rounded_ant * var_ref_ant +
                         (log_var_ant[mask_ant][-1] - slope_rounded_ant * l[mask_ant][-1] - 3))
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure
        
        if result_sl_MC_sweep is not None:
            ax.plot(l, np.log(result_sl_MC_sweep.variances[:result_MLMC.L + 1]) / np.log(result_sl_MC_sweep.N), marker = 'o', linestyle = '-', markersize = 6, linewidth = 3, label = 'Standard MC for P_l', color = 'C1')
            
        ax.plot(l[mask][1:], log_var[1:], marker = 'o', linestyle = '-', markersize = 6, linewidth = 3, label = 'MLMC for P_l - P_{l-1}', color = 'c')
        ax.plot(var_ref, ref_slope, marker = 'o', linestyle = '--', markersize = 4, linewidth = 3, label = f'Slope = {slope_rounded}', color = 'c')
        
        ax.plot(l[mask_ant][1:], log_var_ant[1:], marker = 'o', linestyle = '-', markersize = 6, linewidth = 3, label = r'MLMC for $P^{av}$_l - P_{l-1}', color = 'b')
        ax.plot(var_ref_ant, ref_slope_ant, marker = 'o', linestyle = '--', markersize = 4, linewidth = 3, label = f'Slope = {slope_rounded_ant}', color = 'b')
        
        ax.set_title(f'{result_MLMC.option_name} option, {result_MLMC.scheme_name}', fontsize = 26)
        ax.set_xlabel('Level l', fontsize = 30)
        ax.set_ylabel(r'$\log_N$ Variance', fontsize = 30)
        ax.legend(fontsize = 22)
        ax.tick_params(axis='both', labelsize = 22)
        ax.grid(True)
        
        if ax is None:
            plt.show()
        else:
            return fig, ax
        
    def log_mean_plot(self, result_MLMC, result_MLMC_antithetic, result_sl_MC_sweep = None, ax = None):
        
        # Define semi-log axes
        l = np.arange(result_MLMC.L + 1)
        
        Y_l = result_MLMC.Y_l
        Y_l_ant = result_MLMC_antithetic.Y_l
        
        mask = np.abs(Y_l) > 0
        mask_ant = np.abs(Y_l_ant) > 0
        
        log_Y_l = np.full(len(Y_l), np.nan, dtype = float)
        log_Y_l_ant = np.full(len(Y_l_ant), np.nan, dtype = float)
        
        log_Y_l[mask] = np.log(np.abs(Y_l[mask])) / np.log(result_MLMC.N)
        log_Y_l_ant[mask_ant] = np.log(np.abs(Y_l_ant[mask_ant])) / np.log(result_MLMC_antithetic.N)
        
        if mask.sum() > 3:
            
            # Fit a linear model. Take the last 3 point to capture the true
            # asymptotics and ignore the pre-asymptotics.
            slope, intercept = np.polyfit(l[mask][:2], log_Y_l[mask][:2], deg = 1)
            slope_rounded = round(slope * 2) / 2
            
            # Reference line
            var_ref = np.linspace(1, result_MLMC.L, 50)
            ref_slope = (slope_rounded * var_ref +
                         (log_Y_l[mask][-1] - slope_rounded * l[mask][-1] - 3.5))
            
        if mask_ant.sum() > 3:
            
            # Fit a linear model. Take the last 3 point to capture the true
            # asymptotics and ignore the pre-asymptotics.
            slope_ant, intercept_ant = np.polyfit(l[mask_ant][-3:], log_Y_l_ant[mask_ant][-3:], deg = 1)
            slope_rounded_ant = round(slope_ant * 2) / 2
            
            # Reference line
            var_ref_ant = np.linspace(1, result_MLMC_antithetic.L, 50)
            ref_slope_ant = (slope_rounded_ant * var_ref_ant +
                         (log_Y_l_ant[mask_ant][-1] - slope_rounded_ant * l[mask_ant][-1] - 3))
                
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure
        
        if result_sl_MC_sweep is not None:
            ax.plot(l, np.log(result_sl_MC_sweep.prices[:result_MLMC.L + 1]) / np.log(result_sl_MC_sweep.N), marker = 'o', linestyle = '-', markersize = 6, label = 'Standard MC for P_l', linewidth = 3, color = 'C1')
            
        ax.plot(l[mask][1:], log_Y_l[1:], marker = 'o', linestyle = '-', markersize = 6, label = 'MLMC for P_l - P_{l-1}', color = 'c', linewidth = 3)
        ax.plot(var_ref, ref_slope, marker = 'o', linestyle = '--', markersize = 4, label = f'Slope = {slope_rounded}', color = 'c', linewidth = 3)
        
        ax.plot(l[mask_ant][1:], log_Y_l_ant[1:], marker = 'o', linestyle = '-', markersize = 6, label = r'MLMC for $P^{av}$_l - P_{l-1}', color = 'b', linewidth = 3)
        ax.plot(var_ref_ant, ref_slope_ant, marker = 'o', linestyle = '--', markersize = 4, label = f'Slope = {slope_rounded_ant}', color = 'b', linewidth = 3)
        
        ax.set_title(f'{result_MLMC.option_name} option, {result_MLMC.scheme_name}', fontsize = 26)
        ax.set_xlabel('Level l', fontsize = 30)
        ax.set_ylabel(r'$\log_N$ |Mean|', fontsize = 30)
        ax.legend(fontsize = 22)
        ax.tick_params(axis='both', labelsize = 22)
        ax.grid(True)
        
        if ax is None:
            plt.show()
        else:
            return fig, ax
    
    def samples_plot(self, result_MLMC_eps_antithetic_sweep, ax = None):
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure
        
        for i in range(len(result_MLMC_eps_antithetic_sweep.M_l)):  
            
            l = np.arange(result_MLMC_eps_antithetic_sweep.L[i] + 1)
            M_l = result_MLMC_eps_antithetic_sweep.M_l[i] 
            ax.plot(l, M_l, marker = 'o', linestyle = '-', markersize = 6, linewidth = 3, label = rf'$\epsilon$ = {result_MLMC_eps_antithetic_sweep.eps_set[i]}')

        ax.set_yscale('log')
        ax.set_title(f'{result_MLMC_eps_antithetic_sweep.option_name} option, {result_MLMC_eps_antithetic_sweep.scheme_name}', fontsize = 26)
        ax.set_xlabel('Level l', fontsize = 30)
        ax.set_ylabel(r'Number of samples $M_l$', fontsize = 30)
        ax.legend(fontsize = 22)
        ax.tick_params(axis='both', labelsize = 22)
        ax.grid(True)
        
        if ax is None:
            plt.show()
        else:
            return fig, ax
    
    def complexity_plot(self, result_MLMC_eps_sweep, result_MLMC_eps_antithetic_sweep, result_sl_MC_sweep, ax = None):
        
        epsilon = np.array(result_MLMC_eps_sweep.eps_set)
        epsilon_ant = np.array(result_MLMC_eps_antithetic_sweep.eps_set)
        
        MLMC_cost = np.zeros(len(epsilon))
        MLMC_cost_ant = np.zeros(len(epsilon_ant))
        
        MC_cost = np.zeros(len(epsilon))
                
        N = result_MLMC_eps_sweep.N
        
        for i in range(len(epsilon)):
            
            L = result_MLMC_eps_sweep.L[i]
            L_ant = result_MLMC_eps_antithetic_sweep.L[i]
                        
            M_l = np.array(result_MLMC_eps_sweep.M_l[i])
            M_l_ant = np.array(result_MLMC_eps_antithetic_sweep.M_l[i])
            
            MLMC_cost[i] = M_l[0] * 1
            MLMC_cost_ant[i] = M_l_ant[0] * 1

            
            for l in range(1, L + 1):
                MLMC_cost[i] += M_l[l] * (N**l + N**(l-1))
                
            for l in range(1, L_ant + 1):    
                MLMC_cost_ant[i] += M_l_ant[l] * (2 * N**l + N**(l-1))
             
            M_star = 2 * epsilon[i]**(-2) * result_sl_MC_sweep.variances[:L_ant+1]
            MC_cost[i] = np.sum(M_star * result_sl_MC_sweep.N ** np.arange(L_ant+1))
             
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure

        ax.plot(epsilon, epsilon**2 * MC_cost, marker = 'o', linestyle = '-', markersize = 6, linewidth = 3, label = 'Standard MC', color = 'C1')
        ax.plot(epsilon, epsilon**2 * MLMC_cost_ant, marker = 'o', linestyle = '-', markersize = 6, linewidth = 3, label = 'MLMC antithetic', color = 'b')
        ax.plot(epsilon, epsilon**2 * MLMC_cost, marker = 'o', linestyle = '-', markersize = 6, linewidth = 3, label = 'MLMC', color = 'c')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'{result_MLMC_eps_sweep.option_name} option, {result_MLMC_eps_sweep.scheme_name}', fontsize = 26)
        ax.set_xlabel(r'Accuracy $\epsilon$', fontsize = 30)
        ax.set_ylabel(r'$\epsilon^2$ Cost', fontsize = 30)
        ax.legend(fontsize = 22)
        ax.tick_params(axis='both', labelsize = 22)
        ax.grid(True)
        
        if ax is None:
            plt.show()
        else:
            return fig, ax
        
    def comprehensive_plot(self, result_MLMC, result_MLMC_antithetic, result_MLMC_eps_sweep, result_MLMC_eps_antithetic_sweep, result_sl_MC_sweep):
        
        fig, axes = plt.subplots(2, 2, figsize=(25, 18))
        # fig.suptitle(f'MLMC analysis for {result_MLMC.option_name} option')
        
        self.log_variance_plot(result_MLMC, result_MLMC_antithetic, result_sl_MC_sweep, ax = axes[0, 0])
        self.log_mean_plot(result_MLMC, result_MLMC_antithetic, result_sl_MC_sweep, ax = axes[0, 1])
        self.samples_plot(result_MLMC_eps_antithetic_sweep, ax = axes[1, 0])
        self.complexity_plot(result_MLMC_eps_sweep, result_MLMC_eps_antithetic_sweep, result_sl_MC_sweep, ax = axes[1, 1])
        
        plt.tight_layout()
        plt.show()
        
        
Analysis_MLMC_antithetic = Analysis_antithetic()

Analysis_MLMC_antithetic.comprehensive_plot(European_call_MLMC_heston_run, European_call_MLMC_heston_antithetic_run, European_call_MLMC_heston_eps_sweep, European_call_MLMC_heston_eps_antithetic_sweep, European_call_heston_st_sweep)


Asian_call_heston_st_run = Standard_MC.run_heston(Milstein_Heston, Asian_call_option, 
                                                     S_0 = 100, V_0 = 0.04, M = 10_000, N = 1000)

Asian_call_MLMC_heston_antithetic_run = Multilevel_MC.run_antithetic(Milstein_Heston, Asian_call_option,
                                                             S_0 = 100, V_0 = 0.04, M_in = 100, eps = 0.005,
                                                             N = 2, antithetic = True)

Asian_call_MLMC_heston_run = Multilevel_MC.run_antithetic(Milstein_Heston, Asian_call_option,
                                                             S_0 = 100, V_0 = 0.04, M_in = 100, eps = 0.005,
                                                             N = 2, antithetic = False)

accuracy = [0.1, 0.05, 0.03, 0.01, 0.005, 0.003]
Asian_call_MLMC_heston_eps_antithetic_sweep = Multilevel_MC.run_antithetic_eps_sweep(Milstein_Heston, Asian_call_option,
                                                                             S_0 = 100, V_0 = 0.04, M_in = 100, eps_set = accuracy,
                                                                             N = 2, antithetic = True)

Asian_call_MLMC_heston_eps_sweep = Multilevel_MC.run_antithetic_eps_sweep(Milstein_Heston, Asian_call_option,
                                                                             S_0 = 100, V_0 = 0.04, M_in = 100, eps_set = accuracy,
                                                                             N = 2, antithetic = False)

Asian_call_heston_st_sweep = Standard_MC.run_heston_levels_sweep(Milstein_Heston, Asian_call_option,
                                                                    S_0 = 100, V_0 = 0.04, M = 10_000, L = max(Asian_call_MLMC_heston_run.L, *Asian_call_MLMC_heston_eps_sweep.L))


Analysis_MLMC_antithetic.comprehensive_plot(Asian_call_MLMC_heston_run, Asian_call_MLMC_heston_antithetic_run, Asian_call_MLMC_heston_eps_sweep, Asian_call_MLMC_heston_eps_antithetic_sweep, Asian_call_heston_st_sweep)

