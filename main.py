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
                                                     S_0 = 100, V_0 = 0.04, M = 10_000, N = 1000)

Asian_call_heston_st_run = Standard_MC.run_heston(Milstein_Heston, Asian_call_option, 
                                                     S_0 = 100, V_0 = 0.04, M = 10_000, N = 1000)


# Run the MLMC algorithm and store the output in corresponding objects.
# Parameters: S_0 = 100, M_in = 1000, N = 2, eps is option-specific.
# Warning: if the accuracy is chosen to be not enough small, the algorithm
# may perform only one iteration, providing too few levels to fit a linear model
# which causes an error.
European_call_MLMC_run = Multilevel_MC.run(M_scheme, European_call_option,
                                           S_0 = 100, M_in = 1000, eps = 0.01,
                                           N = 2, bridge = False)

Asian_call_MLMC_run = Multilevel_MC.run(M_scheme, Asian_call_option,
                                        S_0 = 100, M_in = 1000, eps = 0.001,
                                        N = 2, bridge = True)

Lookback_call_MLMC_run = Multilevel_MC.run(M_scheme, Lookback_call_option,
                                           S_0 = 100, M_in = 1000, eps = 0.01,
                                           N = 2, bridge = True)

Digital_call_MLMC_run = Multilevel_MC.run(M_scheme, Digital_call_option,
                                          S_0 = 100, M_in = 1000, eps = 0.00005,
                                          N = 2, bridge = True)

European_call_MLMC_heston_antithetic_run = Multilevel_MC.run_antithetic(Milstein_Heston, European_call_option,
                                                             S_0 = 100, V_0 = 0.04, M_in = 1000, eps = 0.01,
                                                             N = 2, antithetic = True)

European_call_MLMC_heston_run = Multilevel_MC.run_antithetic(Milstein_Heston, European_call_option,
                                                             S_0 = 100, V_0 = 0.04, M_in = 1000, eps = 0.0003,
                                                             N = 2, antithetic = False)

Asian_call_MLMC_heston_antithetic_run = Multilevel_MC.run_antithetic(Milstein_Heston, Asian_call_option,
                                                             S_0 = 100, V_0 = 0.04, M_in = 1000, eps = 0.001,
                                                             N = 2, antithetic = True)

Asian_call_MLMC_heston_run = Multilevel_MC.run_antithetic(Milstein_Heston, Asian_call_option,
                                                             S_0 = 100, V_0 = 0.04, M_in = 1000, eps = 0.001,
                                                             N = 2, antithetic = False)

# Run MLMC_eps_sweep for a set of accuracies.
# Parameters: S_0 = 100, M_in = 1000, N = 2.
accuracy = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
European_call_MLMC_eps_sweep = Multilevel_MC.run_eps_sweep(M_scheme, European_call_option,
                                                           S_0 = 100, M_in = 1000, eps_set = accuracy,
                                                           N = 2, bridge = False)

accuracy = [0.3, 0.1, 0.05, 0.01, 0.005, 0.002]
Asian_call_MLMC_eps_sweep = Multilevel_MC.run_eps_sweep(M_scheme, Asian_call_option,
                                                        S_0 = 100, M_in = 1000, eps_set = accuracy,
                                                        N = 2, bridge = True)

accuracy = [0.3, 0.2, 0.1, 0.05, 0.03, 0.01]
Lookback_call_MLMC_eps_sweep = Multilevel_MC.run_eps_sweep(M_scheme, Lookback_call_option,
                                                           S_0 = 100, M_in = 1000, eps_set = accuracy,
                                                           N = 2, bridge = True)

accuracy = [0.0025, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
Digital_call_MLMC_eps_sweep = Multilevel_MC.run_eps_sweep(M_scheme, Digital_call_option,
                                                          S_0 = 100, M_in = 1000, eps_set = accuracy,
                                                          N = 2, bridge = True)

accuracy = [0.01, 0.005, 0.001, 0.0005, 0.0003, 0.0001]
European_call_MLMC_heston_eps_antithetic_sweep = Multilevel_MC.run_antithetic_eps_sweep(Milstein_Heston, European_call_option,
                                                                             S_0 = 100, V_0 = 0.04, M_in = 1000, eps_set = accuracy,
                                                                             N = 2, antithetic = True)

European_call_MLMC_heston_eps_sweep = Multilevel_MC.run_antithetic_eps_sweep(Milstein_Heston, European_call_option,
                                                                             S_0 = 100, V_0 = 0.04, M_in = 1000, eps_set = accuracy,
                                                                             N = 2, antithetic = False)

accuracy = [0.5, 0.3, 0.1, 0.05, 0.01, 0.005]
Asian_call_MLMC_heston_eps_antithetic_sweep = Multilevel_MC.run_antithetic_eps_sweep(Milstein_Heston, Asian_call_option,
                                                                             S_0 = 100, V_0 = 0.04, M_in = 1000, eps_set = accuracy,
                                                                             N = 2, antithetic = True)

Asian_call_MLMC_heston_eps_sweep = Multilevel_MC.run_antithetic_eps_sweep(Milstein_Heston, Asian_call_option,
                                                                             S_0 = 100, V_0 = 0.04, M_in = 1000, eps_set = accuracy,
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
                                                                    S_0 = 100, V_0 = 0.04, M = 10_000, L = max(European_call_MLMC_heston_antithetic_run.L, *European_call_MLMC_heston_eps_antithetic_sweep.L))

Asian_call_heston_st_sweep = Standard_MC.run_heston_levels_sweep(Milstein_Heston, Asian_call_option,
                                                                    S_0 = 100, V_0 = 0.04, M = 10_000, L = max(Asian_call_MLMC_heston_antithetic_run.L, *Asian_call_MLMC_heston_eps_antithetic_sweep.L))

# Run the Analysis
Analysis_MLMC = mc.Analysis()

Analysis_MLMC.comprehensive_plot(European_call_MLMC_run, European_call_MLMC_eps_sweep, European_call_st_sweep)

Analysis_MLMC.comprehensive_plot(Asian_call_MLMC_run, Asian_call_MLMC_eps_sweep, Asian_call_st_sweep)

Analysis_MLMC.comprehensive_plot(Lookback_call_MLMC_run, Lookback_call_MLMC_eps_sweep, Lookback_call_st_sweep)

Analysis_MLMC.comprehensive_plot(Digital_call_MLMC_run, Digital_call_MLMC_eps_sweep, Digital_call_st_sweep)

Analysis_MLMC.comprehensive_plot(European_call_MLMC_heston_antithetic_run, European_call_MLMC_heston_eps_antithetic_sweep, European_call_heston_st_sweep)

Analysis_MLMC.comprehensive_plot(Asian_call_MLMC_heston_run, Asian_call_MLMC_heston_eps_sweep, Asian_call_heston_st_sweep)