import numpy as np
import pandas as pd
from typing import Tuple


def exponential_smoothing(series, alpha):
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result
  
  
def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series) + 1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result
 


class HoltWinters:
    
    """
    Holt-Winters model with the anomalies detection using Brutlag method
    """
    
    
    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        
        
    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i + self.slen] - self.series[i]) / self.slen
        return sum / self.slen  
    
    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series) / self.slen)
        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen * j:self.slen * j + self.slen]) / float(self.slen))
        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen * j + i] - season_averages[j]
            seasonals[i] = sum_of_vals_over_avg / n_seasons
        return seasonals   

          
    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []
        
        seasonals = self.initial_seasonal_components()
        
        for i in range(len(self.series) + self.n_preds):
            if i == 0: # components initialization
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i % self.slen])
                
                self.PredictedDeviation.append(0)
                
                self.UpperBond.append(self.result[0] + 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                
                self.LowerBond.append(self.result[0] - 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                continue
                
            if i >= len(self.series): # predicting
                m = i - len(self.series) + 1
                self.result.append((smooth + m * trend) + seasonals[i % self.slen])
                
                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1] * 1.01) 
                
            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha * (val - seasonals[i % self.slen]) + (1 - self.alpha) * (smooth + trend)
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i % self.slen] = self.gamma * (val - smooth) + (1 - self.gamma) * seasonals[i % self.slen]
                self.result.append(smooth + trend + seasonals[i % self.slen])
                
                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i]) 
                                               + (1 - self.gamma) * self.PredictedDeviation[-1])
                     
            self.UpperBond.append(self.result[-1] + 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.LowerBond.append(self.result[-1] - 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i % self.slen])

            
class Hedgebeta:
      '''
      Y. Freund, R. Schapire, “A Decision-Theoretic Generalization of on-Line Learning and an Application to Boosting”, 1995.
      '''
      def __init__(self, num_stocks, num_rounds):
        self.total_rsrc = 1.0
        self.wealth = []
        self.total_wealth.append(1.0)

        self.num_stocks = num_stocks
        self.num_rounds = num_rounds

        self.beta = np.sqrt(2.0 * np.log(num_stocks) / num_rounds)
        self.wt = [1.0 / num_stocks] * num_stocks

      def relevance_prices(self, dataframes):
        price_rel_table = np.zeros(shape=(self.num_stocks, self.num_rounds))
        k = 0
        for df in dataframes:
          price_rel_table[k] = np.array([round(df.iloc[i][0] / df.iloc[i][1], 6) \
                                         for i in range(self.num_rounds)])
          k += 1

        max_price_rel = max([max(price_rel_table[k]) for k in range(self.num_stocks)])

        return price_rel_table, max_price_rel

      def fit(self, dataframes):
        price_rel_table, max_price_rel = self.relevance_prices(dataframes)

        for i in range(self.num_rounds):
          allocation = [self.wt[j] * total_rsrc / sum(self.wt) for j in \
                        range(self.num_stocks)]
          current_price_rel_vector = price_rel_table[:, i]

          total_rsrc = np.dot(allocation, current_price_rel_vector)
          total_wealth.append(total_rsrc)

          wt_update = [self.beta ** (max_price_rel - current_price_rel_vector[k]) \
                       for k in range(self.num_stocks)]

          self.wt = [self.wt[j] * wt_update[j] for j in range(self.num_stocks)]
          if min(self.wt) < 0.001:
            self.wt = [self.wt[j] * 1000 for j in range(self.num_stocks)]

      def predict(self):
        pass

    
def nelder_mead(
    f,
    x_start,
    step_for_simplex = 0.1,
    no_improv_break=10,
    max_iter=0,
    alpha=1.,
    gamma=2.,
    rho=-0.5,
    sigma=0.5
) -> Tuple[np.array, float]:
    '''
    https://people.duke.edu/~hpgavin/cee201/Nelder+Mead-ComputerJournal-1965.pdf
    Args:
        f (Callable): function to call
        x_start (np.ndarray): array of shape (n), initial value for algo
        step_for_simplex (float): value for samplex vertices around x_start
        no_improv_break (int): break after no_improv_break iterations 
            without improvement
        max_iter (int): maximum number of iters in algo
        
        alpha (float): reflection param
        gamma (float): expansion param
        rho (float): compression param
        sigma (float): reduction param

    Returns:
        Tuple[np.array, float]: (best parameter array, best score)
    '''
    # init
    space_dim = len(x_start)
    prev_best_score = f(x_start)
    steps_with_no_improv = 0
    simplex = [[x_start, prev_best_score]]
    
    simplex_story = []
    

    # sapmle simplex
    for i in range(space_dim):
        x = x_start.copy()
        x[i] = x[i] + step_for_simplex
        score = f(x)
        simplex.append([x, score])

    # simplex iter
    iters = 0

    pbar = tqdm(range(max_iter))
    for _ in pbar:
        # order
        simplex.sort(key=lambda x: x[1])
        simplex_story.append(copy.copy(simplex))
        
        best_score = simplex[0][1]

        # break after max_iter
        if iters >= max_iter:
            return (*simplex[0], simplex_story)
        iters += 1

        pbar.set_description(f'Best score: {best_score}')

        if best_score < prev_best_score:
            steps_with_no_improv = 0
            prev_best_score = best_score
        else:
            steps_with_no_improv += 1

        if steps_with_no_improv >= no_improv_break:
            return (*simplex[0], simplex_story)

        # centroid
        center_of_mass = [0.] * space_dim
        for x_vector, _ in simplex[:-1]:
            for idx, value in enumerate(x_vector):
                center_of_mass[idx] += value / (len(simplex)-1)

                
        delta_vector = center_of_mass - simplex[-1][0]
        
        # reflection
        x_reflection = center_of_mass + alpha * delta_vector
        reflection_score = f(x_reflection)
        
        if best_score <= reflection_score < simplex[-2][1]:
            del simplex[-1]
            simplex.append([x_reflection, reflection_score])
            continue

        # expansion
        if reflection_score < best_score:
            x_expansion = center_of_mass + gamma * delta_vector
            expansion_score = f(x_expansion)
            if expansion_score < reflection_score:
                del simplex[-1]
                simplex.append([x_expansion, expansion_score])
                continue
            else:
                del simplex[-1]
                simplex.append([x_reflection, reflection_score])
                continue

        # compression
        x_compression = center_of_mass + rho * delta_vector
        compression_score = f(x_compression)
        if compression_score < simplex[-1][1]:
            del simplex[-1]
            simplex.append([x_compression, compression_score])
            continue

        # reduction
        best_x = simplex[0][0]
        new_simplex = [simplex[0]]
        for x_vector, _ in simplex[1:]:
            x_reduction = best_x + sigma * (x_vector - best_x)
            reduction_score = f(x_reduction)
            new_simplex.append([x_reduction, reduction_score])
        simplex = new_simplex
    return simplex
