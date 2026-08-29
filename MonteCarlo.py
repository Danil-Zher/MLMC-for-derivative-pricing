import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import stats
import warnings
from dataclasses import dataclass


class Model:
    '''
    This base class defines parameters for the models used in the project.
    
    Attributes: 
        r - risk-free interest rate;
        sigma - volatility.
    '''
    def __init__(self, r, sigma):
        self.r = r
        self.sigma = sigma

class Black_Scholes(Model):
    '''
    This class sets parameters of the Black-Scholes model:
        dS_t = r S_t dt + sigma S_t dW_t.
    These parameters are used to model the stock price dynamics.
        
    Methods:
        drift - the method returns the drift term of the Black-Scholes SDE;
        diffusion - the method returns the diffusion term of the Black-Scholes SDE;
        diffusion_derivative - the method returns derivative of the diffusion term.
     '''
    def __init__(self, r, sigma):
        super().__init__(r, sigma)
        
    def drift(self, S):
        return self.r * S
    
    def diffusion(self, S):
        return self.sigma * S
    
    def diffusion_derivative(self):
        return self.sigma    
    
class Heston(Model):
    '''
    This class sets parameters of the Heston model:
        dS_t = r S_t dt + sqrt(V_t) S_t dW_1_t
        dV_t = kappa (theta - V_t) dt + sigma sqrt(V_t) dW_2_t
        
        X = log(S)
        
        dX_t = (r - 1/2 V_t) dt + sqrt(V_t) dW_1_t
        dV_t = kappa (theta - V_t) dt + sigma sqrt(V_t) dW_2_t
        
    Attributes: 
        sigma - volatility of volatility;
        kappa - mean-reversion speed;
        theta - long-term mean of volatility.
        
    Methods:
        drift_X - the method returns the drift term of the log-Heston stock price SDE;
        diffusion_X - the method returns the diffusion term of the log-Heston stock price SDE;
        drift_V- the method returns the drift term of the log-Heston volatility SDE;
        diffusion_V - the method returns the diffusion term of the log-Heston volatility SDE;
    '''
    def __init__(self, r, sigma, kappa, theta):
        super().__init__(r, sigma)
        self.kappa = kappa
        self.theta = theta
        
    def drift_X(self, V):
        return (self.r - 1/2 * V)
    
    def diffusion_X(self, V):
        return np.sqrt(V)
    
    def drift_V(self, V):
        return self.kappa * (self.theta - V)
    
    def diffusion_V(self, V):
        return self.sigma * np.sqrt(V)
    

class Scheme:
    '''
    This class defines a base class of approximating schemes
    used in the numerical solution of scalar SDE's.
    
    Attributes:
        model - the model governing the stock price dynamics.
        
    Methods:
        step - the method performs one step of the numerical step, updating
        the state vector S using the given time step dt and
        Brownian motion increments dW.
    '''
    def __init__(self, model):
        self.model = model
        
class Euler_Maruyama_scheme(Scheme):
    '''
    This class provides a method for performing one step
    of the Euler-Maruyama scheme.
    '''
    def step(self, S, dt, dW):
        return S + self.model.drift(S) * dt + self.model.diffusion(S) * dW

class Milstein_scheme(Scheme):
    '''
    This class provides a method for performing one step
    of the Milstein scheme.
    '''
    def step(self, S, dt, dW):
        return (
               S + self.model.drift(S) * dt +
               self.model.diffusion(S) * dW +
               0.5 * self.model.diffusion(S) * 
               self.model.diffusion_derivative() * (dW**2 - dt)
               )
    
    
class log_Heston_Milstein():
    '''
    This class provides a method performing one step
    of the Milstein scheme in the log-Heston framework.
    '''
    def __init__(self, model):
        self.model = model
    
    def step(self, X, V, dt, dW_1, dW_2):
        
        X_new = (X + self.model.drift_X(V) * dt +
                 self.model.diffusion_X(V) * dW_1 +
                 1/4 * self.model.sigma * dW_1 * dW_2)
        
        denomenator_factor = 1 / (1 + self.model.kappa * dt)
        
        V_new = (V + self.model.kappa * self.model.theta * dt +
                 self.model.diffusion_V(V) * dW_2 +
                 1/4 * self.model.sigma**4 * (dW_2**2 - dt)) * denomenator_factor
        
        V_new = np.maximum(V_new, 0)

        return X_new, V_new


class Option:
    '''
    This class defines a base class of options to analyse.
    
    Attributes:
        T - length of the option contract;
        K - strike price of the option (if applicable);
        weak_order - weak convergence rate for the used schemes and
        options under consideration.
        
    Methods:
        statistics - the method initialises price statistics
        to be used for computing the payoff;
        
        update_statistics - the method performs one step of updating the statistics
        pushing it forward the final statistics;
        
        bridge_fine_update - the method computes the Brownian bridge correction term
        on the fine grid and stores the auxiliary values needed for coarse-grid corrections;
        
        bridge_coarse_update - the method computes the Brownian bridge correction term
        on the coarse grid, reusing the auxiliary values produced by bridge_fine_update;
        
        payoff - the method computes the option payoff.
        '''
    def __init__(self, T, K):
        self.T = T
        self.K = K
        self.weak_order = 1

class European_call(Option):
    
    def statistics(self, M, S_0):
        # We only need S_T for computing the payoff of a European call option.
        self.S_T = np.full(M, S_0)
        
    def update_statistics(self, S, dt, model, bridge = False, bridge_term = None):
        self.S_T = S  
        
    def payoff(self):
        return np.maximum(self.S_T - self.K, 0)
        
class Asian_call(Option):
    
    def statistics(self, M, S_0):
        # We only need an integral mean S_mean for computing
        # the payoff of an Asian call option.
        self.S_sum = np.zeros(M)
        self.previous_S = np.full(M, S_0)
            
    def update_statistics(self, S, dt, model, bridge = False, bridge_term = None):
                
        correction = 0 if bridge is False else bridge_term
        
        # Compute the intergal mean using the trapezoidal rule.
        self.S_sum += 0.5 * (S + self.previous_S) * dt + correction
        self.previous_S = S
         
    def bridge_fine_update(self, model, S_prev, S, dt, rng):
        b_n = model.diffusion(S_prev)
        dI = rng.normal(loc = 0, scale = np.sqrt(dt**3 / 12), size = len(S))
        return dI, b_n * dI
    
    def bridge_coarse_update(self, model, S_prev, S, dt, memory_1, memory_2, dW_1, dW_2):
        b_n = model.diffusion(S_prev)
        dI = memory_1 + memory_2 + 0.5 * dt * (dW_1 - dW_2)
        return b_n * dI
        
    def payoff(self):
        return np.maximum(self.S_sum / self.T - self.K, 0)

class Lookback_call(Option):
    
    def __init__(self, T, K):
        super().__init__(T, K)
        # beta_star term improves the weak convergence rate to 1
        self.beta_star = 0.5826
    
    def statistics(self, M, S_0):
        # We need both the final stock price and the minimum value
        # of the stock price over the life of the option
        # for computing the payoff of a Lookback option.
        self.S_min = np.full(M, S_0)
        self.S_T = np.full(M, S_0) 
        
    def update_statistics(self, S, dt, model, bridge = False, bridge_term = None):
        
        # Candidate for S_min before the correction to improve the weak convergence rate
        S_min_cand = S if bridge_term is None else bridge_term
        
        if bridge is not False:
            self.beta_star = 0
        
        self.S_min = np.minimum(self.S_min, S_min_cand - self.beta_star * model.diffusion(S) * np.sqrt(dt))
        self.S_T = S    
        
    def bridge_fine_update(self, model, S_prev, S, dt, rng):
        
        b_n = model.diffusion(S_prev)
        U = rng.uniform(0, 1, size = len(S))
        
        root = np.maximum((S - S_prev)**2 - 2 * b_n ** 2 * dt * np.log(U), 0)
        S_candidate = 0.5 * (S_prev + S - np.sqrt(root))
        
        return U, S_candidate
    
    def bridge_coarse_update(self, model, S_prev, S, dt, memory_1, memory_2, dW_1, dW_2):
        
        b_n = model.diffusion(S_prev)
        D = dW_2 - dW_1
        S_mid = 0.5 * (S_prev + S - b_n * D)
        
        root_1 = np.maximum((S_mid - S_prev)**2 - 2 * b_n**2 * dt * np.log(memory_1), 0)
        root_2 = np.maximum((S - S_mid)**2 - 2 * b_n**2 * dt * np.log(memory_2), 0)
        
        S_cand_1 = 0.5 * (S_prev + S_mid - np.sqrt(root_1))
        S_cand_2 = 0.5 * (S_mid + S - np.sqrt(root_2))
                
        return np.minimum(S_cand_1, S_cand_2)
        
    def payoff(self):
        # A floating strike price is used.
        return (self.S_T - self.S_min)

class Digital_call(Option):
    '''
    bridge_payoff - the method computes the smoothed payoff using the
    conditional expectation technique.
    '''
    def statistics(self, M, S_0):
        # We only need S_T for computing the payoff of a Digital call option.
        self.S_T = np.full(M, S_0)
        
    def update_statistics(self, S, dt, model, bridge = False, bridge_term = None):
        self.S_T = S    
    
    def payoff(self):
        return  np.where(self.S_T >= self.K, 1, 0)
    
    def bridge_payoff(self, model, S_prev, dt, sq_dt, dW = 0):
        
        a_n = model.drift(S_prev)
        b_n = model.diffusion(S_prev)
        
        return stats.norm.cdf((S_prev + a_n * dt + b_n * dW - self.K) / (b_n * sq_dt))
        


@dataclass
class Result():
    '''
    This class classify outcomes of running any Monte Carlo algorithms
    (Single-level or MLMC). It is needed to pass on the information about
    the context of the algorithm run to the downstream Analysis class.
    
    Attributes:
        option - the option under consideration;
        scheme - the numerical scheme used.
    '''
    option: Option
    scheme: Scheme

    @property
    def option_name(self):
        return type(self.option).__name__
    
    @property
    def scheme_name(self):
        return type(self.scheme).__name__

@dataclass
class Single_level_MC_result(Result):
    '''
    This class holds the outcome of one Single_level_MC.run() and
    Single_level_MC.run_heston() call.
    
    Attributes:
        price - the standard Monte Carlo estimate of the option price;
        price_std - the standard deviation of the estimated option price;
        std - the standard deviation of a single sample.
    '''
    price: float
    price_std: float
    std: float

@dataclass
class Single_level_MC_sweep_result(Result):
    '''
    This class holds the outcome of one Single_level_MC.run_levels_sweep() and
    Single_level_MC.run_heston_levels_sweep() call. 
    
    Attributes:
        prices - list of price estimates P_l, one per level;
        variances - list of sample variances Var[P_l] of a single sample;
        N - refinement factor;
        L - the finest level.
    '''
    prices: np.ndarray
    variances: np.ndarray
    N: int
    L: int

@dataclass
class MLMC_result(Result):
    '''
    This class holds the outcome of one MLMC.run() and MLMC.run_antithetic() call.
    
    Attributes:
        price - the MLMC estimate of the option price;
        price_std - the standard deviation of the estimated option price;
        Y_l - list of MC estimates of the correction terms Y_l;
        var - list of sample variances Var[P_l - P_{l-1}];
        M_sim - list of numbers of simulations performed level-wise;
        L - the finest level;
        N - refinement factor;
        eps - required RMSE error.
    '''
    price: float
    price_std: float
    Y_l: np.ndarray
    var: np.ndarray
    M_sim: np.ndarray
    L: int
    N: int
    eps: float
    antithetic: bool = None
     
@dataclass
class MLMC_eps_sweep_result(Result):
    '''
    This class holds the outcome of one MLMC.run_eps_sweep() or
    MLMC.run_heston_levels_sweep() call.
    
    Attributes:
        M_l - list of arrays of number of samples level-wise,
        one array for one eps in eps_set;
        L - list of the finest levels reached, one per eps in eps_set;
        N - refinement factor
        eps_set - array of RMSE errors to sweep over.
    '''
    M_l: list
    L: list
    N: int
    eps_set: list
    antithetic: bool = None
    
    
class Single_level_MC:
    '''
    This class defines the Single-level Monte Carlo algorithm.
    
    Methods:
        run - the method performs one run of the algorithm for
        the Blach-Schoels model and returns
        the estimated option price and its standard deviation;
        
        Parameters:
            scheme - the numerical scheme used;
            option - the option under consideration;
            S_0 - initial stock price;
            M - number of samples for the Monte Carlo estimator;
            N - total number of time steps used in the numerical scheme.
            
        run_levels_sweep - the method performs a sequence of single-level
        Monte Carlo algorithm runs for different grids - with discretisation
        step N^l, l = 0,1,...,L.
        It returns an object of the Single_level_MC_sweep_result class.
        
        Parameters:
            N - refinement factor;
            L - the number of different time discretisation steps used
            in the numerical scheme.
            The total number of different grids is (L+1).
            
         run_heston - the methid performs one run of the algorithm for
         the log-Heston model and returns the estimated option price and
         its standard deviation.
         
         Parameters:
             V_0 - initial volatility;
             N - total number of timesteps used in the numerical scheme.
             
        run_heston_levels_sweep - the method performs a aequence of single-level
        Monte Carlo algorithm runs for the log-Heston model on fidderents grids.
        It returns an object of the Single_level_MC_sweep_result class.
        
        Parameters:
            N - refinement factor.
    '''
    def run(self, scheme, option, S_0, M, N, bridge = False):

        rng = np.random.default_rng()
        
        # Define the time step
        dt = option.T / N
        sq_dt = np.sqrt(dt)
        
        # Initialise the stock price
        S = np.full(M, S_0)
        
        option.statistics(M, S_0)
                
        for i in range(N):
                        
            dW = sq_dt * rng.normal(loc = 0, scale = 1, size = M)
            
            S_prev = S
            S = scheme.step(S, dt, dW)
            
            if bridge is not False and not hasattr(option, 'bridge_payoff'):
                _, correction_term = option.bridge_fine_update(scheme.model, S_prev, S, dt, rng)
                option.update_statistics(S, dt, scheme.model, bridge, bridge_term = correction_term)
            else:
                option.update_statistics(S, dt, scheme.model)
        
        # Compute the vector of discounted option payoffs
        if bridge is not False and hasattr(option, 'bridge_payoff'):
            V = np.exp(- scheme.model.r * option.T) * option.bridge_payoff(scheme.model, S_prev, dt, sq_dt, dW = 0)
        else:
            V = np.exp(- scheme.model.r * option.T) * option.payoff()
        
        a_M = np.mean(V)
        b_M = np.std(V, ddof = 1)
        
        price_std = b_M / np.sqrt(M)
            
        return Single_level_MC_result(option = option,
                                      scheme = scheme,
                                      price = a_M,
                                      price_std = price_std,
                                      std = b_M)
    
    def run_levels_sweep(self, scheme, option, S_0, M, L, N = 2, bridge = False):
        
        # Lists of prices and their sample variances
        prices = np.zeros(L + 1)
        variances = np.zeros(L + 1)
                        
        for l in range(L + 1):
            
            res = self.run(scheme, option, S_0, M, N**l, bridge)
            
            prices[l] = res.price
            variances[l] = res.std ** 2

        return Single_level_MC_sweep_result(option = option,
                                            scheme = scheme,
                                            prices = prices,
                                            variances = variances,
                                            N = N,
                                            L = L)    

    def run_heston(self, scheme, option, S_0, V_0, M, N):
        
        rng = np.random.default_rng()
        
        # Define the time step
        dt = option.T / N
        sq_dt = np.sqrt(dt)
        
        # Initialise the log-price and volatility
        X = np.full(M, np.log(S_0))
        V = np.full(M, V_0)
        
        option.statistics(M, S_0)
        
        for i in range(N):
            
            dW_1 = sq_dt * rng.normal(loc = 0, scale = 1, size = M)
            dW_2 = sq_dt * rng.normal(loc = 0, scale = 1, size = M)
            
            X, V = scheme.step(X, V, dt, dW_1, dW_2)
            
            option.update_statistics(np.exp(X), dt, scheme.model)
        
        # Compute the vector of discounted option payoffs
        P = np.exp(- scheme.model.r * option.T) * option.payoff()
        
        a_M = np.mean(P)
        b_M = np.std(P, ddof = 1)
        
        price_std = b_M / np.sqrt(M)
        
        return Single_level_MC_result(option = option,
                                      scheme = scheme,
                                      price = a_M,
                                      price_std = price_std,
                                      std = b_M)
    
    def run_heston_levels_sweep(self, scheme, option, S_0, V_0, M, L, N = 2):
        
        # Lists of prices and their sample variances
        prices = np.zeros(L + 1)
        variances = np.zeros(L + 1)
                        
        for l in range(L + 1):
            
            res = self.run_heston(scheme, option, S_0, V_0, M, N**l)
            
            prices[l] = res.price
            variances[l] = res.std ** 2

        return Single_level_MC_sweep_result(option = option,
                                            scheme = scheme,
                                            prices = prices,
                                            variances = variances,
                                            N = N,
                                            L = L)


class MLMC:
    '''
    This class defines the Multi-level Monte Carlo algorithm.
    
    Methods:
        run - the method performs one run of the algorithm and returns
        the estimated option price, its standard deviation, the vector of
        of MC estimators of corrections, the vector of their sample variances,
        and the number of performed simulations level-wise.
        
        Parameters:
            scheme - the numerical scheme used;
            option - the option under consideration;
            S_0 - initial stock price;
            M_in - initial allocation of number of samples
            for Monte Carlo estimators of the correction terms Y_l.
            The same number M_in is used for all level
            to run the first iteration;
            N - refinement factor;
            eps - required RMSE error:
            bridge - flag variable that enables Brownian bridge technique.
            
        run_eps_sweep - the method performs a sequence of MLMC
        algorithm runs for different target RMSE errors listed in eps_set.
        It returns an object of the MLMC_eps_sweep_result class.
        
        Parameters:
            eps_set - array of RMSE errors to sweep over.
            
        run_antithetic - the method performs one run of the MLMC algorithm
        for the log-Heston model, optionally using the antithetic approach.
        The method is adapted specifically for the case N = 2.
        It returns an object of the MLMC_result class.
        
        Parameters:
            V_0 - initial volatility;
            antithetic - flag that enables the anthitetic calculations.
            
        run_antithetic_eps_sweep - the method performs a sequence of
        run_antithetic calls for different accuracies eps. It returns an
        object of the MLMC_eps_sweep_result class.
    '''
    def run(self, scheme, option, S_0, M_in, eps, N = 2, bridge = False):
        
        if N != 2 and bridge is not False:
            raise ValueError("The method supports Brownian bridges "
                             "only for the case N = 2")
        
        rng = np.random.default_rng()
        
        # Maximum value of L with a margin
        L_max = 20
        
        # Initial value of L
        L = 1
        
        # Create 2 copies of the option object - one for the fine grid,
        # the second one for the coarse grid
        option_fine = copy.deepcopy(option)
        option_coarse = copy.deepcopy(option)
        
        # Cumulative sums of the samples over all iterations for all levels
        Y_sum = np.zeros(L_max + 1)
        
        # Cumulative sums of squares of the samples
        # over all iterations for all levels
        Y_squared_sum = np.zeros(L_max + 1)

        # Estimates Y_l at the current iteration for all levels
        Y_l = np.zeros(L_max + 1)

        # Current (at given iteration) sample variances
        # of MC estimators at given level l
        var = np.zeros(L_max + 1)
        
        # Optimal number of samples at level l
        M_l_opt = np.full(L_max + 1, M_in)
        
        # Number of performed simulation at level l.
        # It takes into account all iterations.
        M_sim = np.zeros(L_max + 1, dtype = np.int64)
        
        # Discretisation steps
        dt = option.T / N**np.arange(L_max + 1)
        sq_dt = np.sqrt(dt)
        
        # Initial value of the bias chosen randomly large
        # to enter the following loop
        bias = np.inf
        
        # Loop responsible for the bias convergence
        while (L < L_max) and (bias >= 1/np.sqrt(2) * (N ** option.weak_order - 1) * eps):
            
            L += 1
            
            # Loop responsible for the variance convergence
            while np.any(M_sim[:L+1] < M_l_opt[:L+1]):
                
                # Calculation of all Y_l individually
                for l in range(L + 1):
                    
                    if M_sim[l] >= M_l_opt[l]:
                        continue
                    
                    # Number of samples to be additionally computed
                    # at j_th iteration to obtain the
                    # optimal number of samples
                    delta = M_l_opt[l] - M_sim[l]
                                        
                    # Stock prices at time t = 0 computed using the approximating scheme
                    # with N^l and N^(l-1) discretisation steps
                    S = np.full(delta, S_0)
                    S_1 = np.full(delta, S_0)
                    
                    option_fine.statistics(delta, S_0)
                    
                    # Brownian motion increments on the finer grid.
                    # Each column is one Brownian motion path. 
                    dW = sq_dt[l] * rng.normal(loc = 0, scale = 1, size = (int(N**l), delta))
                    
                    if bridge is not False:
                        memory_fine = np.zeros((N**l, delta))
                    
                    for i in range(N**l):
                        
                        S_prev = S
                        S = scheme.step(S, dt[l], dW[i])
                        
                        if bridge is not False and not hasattr(option, 'bridge_payoff'):
                            memory_fine[i], correction_term = option_fine.bridge_fine_update(scheme.model,
                                                                                             S_prev, S, dt[l], rng)
                            option_fine.update_statistics(S, dt[l], scheme.model, bridge, bridge_term = correction_term)
                        else:
                            option_fine.update_statistics(S, dt[l], scheme.model)
                    
                    # Compute the vector of the option payoffs on the finest grid
                    if bridge is not False and hasattr(option, 'bridge_payoff'):
                        P = (np.exp(- scheme.model.r * option.T) *
                             option_fine.bridge_payoff(scheme.model, S_prev, dt[l], sq_dt[l], dW = 0))
                    else:
                        P = np.exp(- scheme.model.r * option.T) * option_fine.payoff()   
                            
                    if l == 0:
                        diff = P
                    else:    
                        
                        option_coarse.statistics(delta, S_0)
                        
                        # Brownian motion on the coarser grid.
                        # Note that the Brownian motion incremets' variance property
                        # is satisfied automatically by construction.
                        dW_1 = dW.reshape(N**(l-1), N, delta).sum(axis = 1)
                        
                        for i in range(N**(l-1)):
                            
                            S_1_prev = S_1
                            S_1 = scheme.step(S_1, dt[l-1], dW_1[i])
                            
                            if bridge is not False and not hasattr(option, 'bridge_payoff'):
                                bridge_coarse = option_coarse.bridge_coarse_update(scheme.model,
                                                                               S_1_prev, S_1, dt[l],
                                                                               memory_fine[2*i], memory_fine[2*i+1],
                                                                               dW[2*i], dW[2*i+1])
                                option_coarse.update_statistics(S_1, dt[l-1], scheme.model, bridge, bridge_term = bridge_coarse)
                            else:
                                option_coarse.update_statistics(S_1, dt[l-1], scheme.model)
                            
                        # Compute the vector of the option payoffs on the cosrser grid
                        if bridge is not False and hasattr(option, 'bridge_payoff'):
                            P_1 = (np.exp(- scheme.model.r * option.T) *
                                   option_coarse.bridge_payoff(scheme.model, S_1_prev, dt[l-1], sq_dt[l], dW[-2]))
                        else:
                            P_1 = np.exp(- scheme.model.r * option.T) * option_coarse.payoff()
                        
                        # Compute correstion terms
                        diff = P - P_1
                        
                    Y_sum[l] += np.sum(diff)
                    Y_squared_sum[l] += np.sum(diff**2)
                    
                    M_sim[l] += delta
                    
                    Y_l[l] = Y_sum[l] / M_sim[l]
                    var[l] = (Y_squared_sum[l] - M_sim[l] * Y_l[l]**2) / (M_sim[l] - 1)
                
                # The copy of the var array that affects only the sample allocation.
                # It is neede to handle the case of zero-variance estimates.
                var_allocation = var[:L+1].copy()
                positive_var = var_allocation > 0
                
                # Replace zero variances by their interpolated values
                if positive_var.sum() >= 2 and not positive_var.all():
                     level = np.arange(L+1)[positive_var]
                     slope, intercept = np.polyfit(level, np.log(var_allocation[positive_var]), deg = 1)
                     zero_var = ~positive_var
                     var_allocation[zero_var] = np.exp(intercept + slope * np.arange(L+1)[zero_var])
                     
                # Compute optimal number of samples level-wise    
                A = np.sum(np.sqrt(var_allocation[:L+1] / dt[:L+1]))
                
                M_l_opt[:L+1] = np.ceil((2 * eps**(-2) * np.sqrt(var_allocation[:L+1] * dt[:L+1])) * A).astype(int)
                
            # Recalculate the bias                
            bias = np.maximum(N ** (- option.weak_order) * np.abs(Y_l[L-1]), np.abs(Y_l[L]))
                
            if L == L_max:
                warnings.warn("Maximum level reached before bias convergence")
                break
        
        # The final estimate of the option price
        Y_hat = np.sum(Y_l[:L+1])
        
        # The standard deviation of the final price estimate
        Y_l_std = np.sqrt(np.sum(var[:L+1] / M_sim[:L+1]))

        return MLMC_result(option = option,
                           scheme = scheme,
                           price = Y_hat,
                           price_std = Y_l_std,
                           Y_l = Y_l[:L+1],
                           var = var[:L+1],
                           M_sim = M_sim[:L+1],
                           L = L,
                           N = N,
                           eps = eps,
                           antithetic = None)
    
    def run_eps_sweep(self, scheme, option, S_0, M_in, eps_set, N = 2, bridge = False):
        
        # List of arrays of number of samples level-wise
        M_l = []
        
        # List of the finest levels reached, one per eps in eps_set
        L = []
        
        for i in range(len(eps_set)):
            
            res = self.run(scheme, option, S_0, M_in, eps_set[i], N, bridge)
        
            M_l.append(res.M_sim)
            
            L.append(res.L)
        
        return MLMC_eps_sweep_result(option = option,
                                     scheme = scheme,
                                     M_l = M_l,
                                     L = L,
                                     N = N,
                                     eps_set = eps_set,
                                     antithetic = None)    

    def run_antithetic(self, scheme, option, S_0, V_0, M_in, eps, N = 2, antithetic = True):
        
        if N != 2:
            raise ValueError("Antithetic approach requires N = 2.")
        
        rng = np.random.default_rng()
        
        # Maximum value of L with a reserve
        L_max = 20
        
        # Initial value of L
        L = 1
        
        # Create 2 copies of the option object - one for the fine grid,
        # the second one for the coarse grid
        option_f = copy.deepcopy(option)
        option_c = copy.deepcopy(option)
        if antithetic:
            option_a = copy.deepcopy(option)
        
        # Cumulative sums of the samples over all iterations for all levels
        Y_sum = np.zeros(L_max + 1)
        
        # Cumulative sums of squares of the samples
        # over all iterations for all levels
        Y_squared_sum = np.zeros(L_max + 1)

        # Estimators Y_l at the current iteration for all levels
        Y_l = np.zeros(L_max + 1)

        # Current (at given iteration) sample variances
        # of MC estimators at given level l
        var = np.zeros(L_max + 1)
        
        # Optimal number of samples at level l
        M_l_opt = np.full(L_max + 1, M_in)
        
        # Number of performed simulation at level l.
        # It takes into account all iterations.
        M_sim = np.zeros(L_max + 1, dtype = np.int64)
        
        # Discretisation steps
        dt = option.T / N**np.arange(L_max + 1)
        sq_dt = np.sqrt(dt)
        
        # Initial value of the bias chosen randomly large
        # to enter the following loop
        bias = np.inf
        
        # Loop responsible for the bias convergence
        while (L < L_max) and (bias >= 1/np.sqrt(2) * (N ** option.weak_order - 1) * eps):
            
            L += 1
            
            # Loop responsible for the variance convergence
            while np.any(M_sim[:L+1] < M_l_opt[:L+1]):
                
                # Calculation of all Y_l individually
                for l in range(L + 1):
                    
                    if M_sim[l] >= M_l_opt[l]:
                        continue
                    
                    # Number of samples to be additionally computed
                    # at j_th iteration to obtain the
                    # optimal number of samples
                    delta = M_l_opt[l] - M_sim[l]
                                  
                    # Log-price and volatility for the fine-grid approximation.
                    # Initial values at t = 0
                    X0 = np.full(delta, np.log(S_0))
                    V0 = np.full(delta, V_0)
                    
                    if l == 0:
                        # Perform one step of updating the option statistics
                        # for the coarsest grid
                        option_f.statistics(delta, S_0)
                    
                        # Two independent Brownian motion increments for both processes
                        dW_1 = sq_dt[0] * rng.normal(loc = 0, scale = 1, size = delta)
                        dW_2 = sq_dt[0] * rng.normal(loc = 0, scale = 1, size = delta)
                        
                        X, V = scheme.step(X0, V0, dt[0], dW_1, dW_2)
                        
                        option_f.update_statistics(np.exp(X), dt[0], scheme.model)
                        
                        diff = np.exp(- scheme.model.r * option.T) * option_f.payoff()
                        
                    else:
                        
                        option_f.statistics(delta, S_0)
                        option_c.statistics(delta, S_0)
                        
                        if antithetic:
                            option_a.statistics(delta, S_0)
                        
                        # Independent Brownian motion increments W_i_j,
                        # where i is an index of a process and j is an index of a timestep
                        dW_1_1 = sq_dt[l] * rng.normal(loc = 0, scale = 1, size = (N**(l-1), delta))
                        dW_1_2 = sq_dt[l] * rng.normal(loc = 0, scale = 1, size = (N**(l-1), delta))
                        dW_2_1 = sq_dt[l] * rng.normal(loc = 0, scale = 1, size = (N**(l-1), delta))
                        dW_2_2 = sq_dt[l] * rng.normal(loc = 0, scale = 1, size = (N**(l-1), delta))
                        
                        X_f, V_f = X0.copy(), V0.copy()
                        X_c, V_c = X0.copy(), V0.copy()
                        
                        if antithetic:
                            X_a, V_a = X0.copy(), V0.copy()
                        
                        for i in range(N**(l-1)):
                            
                            # Perform two sequential steps for each X_f, V_f, X_a, V_a
                            X_f, V_f = scheme.step(X_f, V_f, dt[l], dW_1_1[i], dW_2_1[i])
                            option_f.update_statistics(np.exp(X_f), dt[l], scheme.model)
                            X_f, V_f = scheme.step(X_f, V_f, dt[l], dW_1_2[i], dW_2_2[i])
                            option_f.update_statistics(np.exp(X_f), dt[l], scheme.model)
                            
                            if antithetic:                            
                                X_a, V_a = scheme.step(X_a, V_a, dt[l], dW_1_2[i], dW_2_2[i])
                                option_a.update_statistics(np.exp(X_a), dt[l], scheme.model)
                                X_a, V_a = scheme.step(X_a, V_a, dt[l], dW_1_1[i], dW_2_1[i])
                                option_a.update_statistics(np.exp(X_a), dt[l], scheme.model)
                            
                            # Two fine-grid steps combined together give one coarse-grid step
                            X_c, V_c = scheme.step(X_c, V_c, dt[l-1], dW_1_1[i] + dW_1_2[i], dW_2_1[i] + dW_2_2[i])
                            option_c.update_statistics(np.exp(X_c), dt[l-1], scheme.model)
                            
                        # Compute the vector of the option payoffs 
                        P_f = np.exp(- scheme.model.r * option.T) * option_f.payoff()
                        P_c = np.exp(- scheme.model.r * option.T) * option_c.payoff()
                        
                        if antithetic:
                            P_a = np.exp(- scheme.model.r * option.T) * option_a.payoff()
                            diff = 1/2 * (P_f + P_a) - P_c
                        else:
                            diff = P_f - P_c
                            
                    Y_sum[l] += np.sum(diff)
                    Y_squared_sum[l] += np.sum(diff**2)
                    
                    M_sim[l] += delta
                    
                    Y_l[l] = Y_sum[l] / M_sim[l]
                    var[l] = (Y_squared_sum[l] - M_sim[l] * Y_l[l]**2) / (M_sim[l] - 1)
                
                # The copy of the var array that affects only the sample allocation.
                # It is neede to handle the case of zero-variance estimates.
                var_allocation = var[:L+1].copy()
                positive_var = var_allocation > 0
                
                # Replace zero variances by their interpolated values
                if positive_var.sum() >= 2 and not positive_var.all():
                     level = np.arange(L+1)[positive_var]
                     slope, intercept = np.polyfit(level, np.log(var_allocation[positive_var]), deg = 1)
                     zero_var = ~positive_var
                     var_allocation[zero_var] = np.exp(intercept + slope * np.arange(L+1)[zero_var])
                     
                # Compute optimal number of samples level-wise    
                A = np.sum(np.sqrt(var_allocation[:L+1] / dt[:L+1]))
                
                M_l_opt[:L+1] = np.ceil((2 * eps**(-2) * np.sqrt(var_allocation[:L+1] * dt[:L+1])) * A).astype(int)
                
            # Recalculate the bias                
            bias = np.maximum(N ** (- option.weak_order) * np.abs(Y_l[L-1]), np.abs(Y_l[L]))
                
            if L == L_max:
                warnings.warn("Maximum level reached before bias convergence")
                break
        
        # The final estimate of the option price
        Y_hat = np.sum(Y_l[:L+1])
        
        # The standard deviation of the final price estimate
        Y_l_std = np.sqrt(np.sum(var[:L+1] / M_sim[:L+1]))

        return MLMC_result(option = option,
                           scheme = scheme,
                           price = Y_hat,
                           price_std = Y_l_std,
                           Y_l = Y_l[:L+1],
                           var = var[:L+1],
                           M_sim = M_sim[:L+1],
                           L = L,
                           N = N,
                           eps = eps,
                           antithetic = antithetic)    

    def run_antithetic_eps_sweep(self, scheme, option, S_0, V_0, M_in, eps_set, N = 2, antithetic = True):
        
        # List of arrays of number of samples level-wise
        M_l = []
        
        # List of the finest levels reached, one per eps in eps_set
        L = []
        
        for i in range(len(eps_set)):
            
            res = self.run_antithetic(scheme, option, S_0, V_0,
                                      M_in, eps_set[i], N, antithetic)
        
            M_l.append(res.M_sim)
            
            L.append(res.L)
        
        return MLMC_eps_sweep_result(option = option,
                                     scheme = scheme,
                                     M_l = M_l,
                                     L = L,
                                     N = N,
                                     eps_set = eps_set,
                                     antithetic = antithetic)  
        

class Analysis:
    '''
    This class provides functional for analysis of the results obtained
    using the Single-level and Multi-level Monte Carlo algorithms.
    
    Methods:
        output - the method prints the estimated option price and
        its standard deviation;
        
        log_variance_plot - the method plots log_N(Var[P_l - P_{l-1}]) vs level l
        for the MLMC result, fits a linear model to estimate the slope, and
        plots an equivalent plot for the standard MC estimator;
        
        log_mean_plot - the method plots log_N(E[P_l - P_{l-1}]) vs level l
        for the MLMC result, fits a linear model to estimate the slope, and
        plots an equivalent plot for the standard MC estimator;
        
        samples_plot - the method plots the number of samples M_l required
        level-wise fot the MLMC algorithm, for each target accuracy in a given set;
        
        complexity_plot - the method plots eps^2 * Cost vs the target accuracy
        for both the MLMC and the standard MC algorithms;
        
        comprehensive_plot - the method combines all previously nited plots into
        a single figure with four subplots.
    '''
    def output(self, result):
        print(f'Price of the {result.option_name}: {result.price:.5f}')
        print(f'Standard deviation of the {result.option_name} price: {result.price_std:.5f}')

    def log_variance_plot(self, result_MLMC, result_sl_MC_sweep = None, ax = None):
        
        # Define semi-log axes
        l = np.arange(result_MLMC.L + 1)
        
        var_l = result_MLMC.var
        mask = var_l > 0
        log_var = np.full(len(var_l), np.nan, dtype = float)
        log_var[mask] = np.log(var_l[mask]) / np.log(result_MLMC.N)
                
        if mask.sum() >= 3:
            
            # Fit a linear model. Take the last 3 point to capture the true
            # asymptotics and ignore the pre-asymptotics.
            slope, intercept = np.polyfit(l[mask][-3:], log_var[mask][-3:], deg = 1)
            slope_rounded = round(slope * 2) / 2
            
            # Reference line
            var_ref = np.linspace(1, result_MLMC.L, 50)
            ref_slope = (slope_rounded * var_ref +
                         (log_var[mask][-1] - slope_rounded * l[mask][-1] - 3))
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure
        
        if result_sl_MC_sweep is not None:
            ax.plot(l, np.log(result_sl_MC_sweep.variances[:result_MLMC.L + 1]) / np.log(result_sl_MC_sweep.N), marker = 'o', linestyle = '-', markersize = 6, linewidth = 3, label = 'Standard MC for P_l', color = 'C1')
        ax.plot(l[mask][1:], log_var[1:], marker = 'o', linestyle = '-', markersize = 6, linewidth = 3, label = 'MLMC for P_l - P_{l-1}', color = 'b')
        ax.plot(var_ref, ref_slope, marker = 'o', linestyle = '--', markersize = 4, linewidth = 3, label = f'Slope = {slope_rounded}', color = 'b')
        
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
        
    def log_mean_plot(self, result_MLMC, result_sl_MC_sweep = None, ax = None):
        
        # Define semi-log axes
        l = np.arange(result_MLMC.L + 1)
        
        Y_l = result_MLMC.Y_l
        mask = np.abs(Y_l) > 0
        log_Y_l = np.full(len(Y_l), np.nan, dtype = float)
        log_Y_l[mask] = np.log(np.abs(Y_l[mask])) / np.log(result_MLMC.N)
        
        if mask.sum() > 3:
            
            # Fit a linear model. Take the last 3 point to capture the true
            # asymptotics and ignore the pre-asymptotics.
            slope, intercept = np.polyfit(l[mask][-3:], log_Y_l[mask][-3:], deg = 1)
            slope_rounded = round(slope * 2) / 2
            
            # Reference line
            var_ref = np.linspace(1, result_MLMC.L, 50)
            ref_slope = (slope_rounded * var_ref +
                         (log_Y_l[mask][-1] - slope_rounded * l[mask][-1] - 3))
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure
        
        if result_sl_MC_sweep is not None:
            ax.plot(l, np.log(result_sl_MC_sweep.prices[:result_MLMC.L + 1]) / np.log(result_sl_MC_sweep.N), marker = 'o', linestyle = '-', markersize = 6, label = 'Standard MC for P_l', linewidth = 3, color = 'C1')
        ax.plot(l[mask][1:], log_Y_l[1:], marker = 'o', linestyle = '-', markersize = 6, label = 'MLMC for P_l - P_{l-1}', color = 'b', linewidth = 3)
        ax.plot(var_ref, ref_slope, marker = 'o', linestyle = '--', markersize = 4, label = f'Slope = {slope_rounded}', color = 'b', linewidth = 3)
        
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
    
    def samples_plot(self, result_MLMC_eps_sweep, ax = None):
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure
        
        for i in range(len(result_MLMC_eps_sweep.M_l)):  
            
            l = np.arange(result_MLMC_eps_sweep.L[i] + 1)
            M_l = result_MLMC_eps_sweep.M_l[i] 
            ax.plot(l, M_l, marker = 'o', linestyle = '-', markersize = 6, linewidth = 3, label = rf'$\epsilon$ = {result_MLMC_eps_sweep.eps_set[i]}')

        ax.set_yscale('log')
        ax.set_title(f'{result_MLMC_eps_sweep.option_name} option, {result_MLMC_eps_sweep.scheme_name}', fontsize = 26)
        ax.set_xlabel('Level l', fontsize = 30)
        ax.set_ylabel(r'Number of samples $M_l$', fontsize = 30)
        ax.legend(fontsize = 22)
        ax.tick_params(axis='both', labelsize = 22)
        ax.grid(True)
        
        if ax is None:
            plt.show()
        else:
            return fig, ax
    
    def complexity_plot(self, result_MLMC_eps_sweep, result_sl_MC_sweep, ax = None):
        
        epsilon = np.array(result_MLMC_eps_sweep.eps_set)
        
        MLMC_cost = np.zeros(len(epsilon))
        MC_cost = np.zeros(len(epsilon))
                
        N = result_MLMC_eps_sweep.N
        
        for i in range(len(epsilon)):
            
            L = result_MLMC_eps_sweep.L[i]
                       
            M_l = np.array(result_MLMC_eps_sweep.M_l[i])
            
            MLMC_heston_factor = 2 if isinstance(result_MLMC_eps_sweep.scheme.model, log_Heston_Milstein) else 1
            MC_heston_factor = 2 if isinstance(result_sl_MC_sweep.scheme.model, log_Heston_Milstein) else 1

            antithetic_factor = 2 if result_MLMC_eps_sweep.antithetic else 1

            MLMC_cost[i] = M_l[0] * MLMC_heston_factor
                    
            for l in range(1, L + 1):
                MLMC_cost[i] += MLMC_heston_factor * M_l[l] * (antithetic_factor * N**l + N**(l-1))    
            
            # Reproduces an iterative search of optimal parameters for the standard Monte Carlo
            M_star = 2 * epsilon[i]**(-2) * result_sl_MC_sweep.variances[:L+1]
            MC_cost[i] = MC_heston_factor * np.sum(M_star * result_sl_MC_sweep.N ** np.arange(L+1))
             
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure

        ax.plot(epsilon, epsilon**2 * MC_cost, marker = 'o', linestyle = '-', markersize = 6, linewidth = 3, label = 'Standard MC', color = 'C1')
        ax.plot(epsilon, epsilon**2 * MLMC_cost, marker = 'o', linestyle = '-', markersize = 6, linewidth = 3, label = 'MLMC', color = 'b')

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
        
    def comprehensive_plot(self, result_MLMC, result_MLMC_eps_sweep, result_sl_MC_sweep):
        
        fig, axes = plt.subplots(2, 2, figsize=(25, 18))
        # fig.suptitle(f'MLMC analysis for {result_MLMC.option_name} option')
        
        self.log_variance_plot(result_MLMC, result_sl_MC_sweep, ax = axes[0, 0])
        self.log_mean_plot(result_MLMC, result_sl_MC_sweep, ax = axes[0, 1])
        self.samples_plot(result_MLMC_eps_sweep, ax = axes[1, 0])
        self.complexity_plot(result_MLMC_eps_sweep, result_sl_MC_sweep, ax = axes[1, 1])
        
        plt.tight_layout()
        plt.show()