import numpy as np
import pandas as pd
import cec2017

from cec2017.functions import f2
from cec2017.functions import f13
from colorama import Fore

# Function 2 from CEC 2017
def q2(x):
    return f2(x)

# Function 13 from CEC 2017
def q13(x):
    return f13(x)

def find_best(P, o):
    '''
    Method that finds the best individual (minimization - smallest objective function result)
    :param P: Population
    :param o: evaluation of population vector
    :return x_best, o_best: Returns the best solution and its evaluation from the given population
    '''
    # Take the argument number of the best evaluation from the evaluation matrix
    o_best_arg = np.argmin(o)
    # Set the best evaluation and the best point
    o_best = o[o_best_arg]
    x_best = P[o_best_arg]
    return x_best, o_best

def stop(t, t_max):
    '''
    Stop criterion
    :param t: iteration number
    :param t_max: number of iterations
    '''
    if t > t_max:
        return True

def evaluate(q, P):
    '''
    Operation of evaluating the function on a given population
    :param q: evaluation function
    :param P: Population
    :return o: evaluation of population vector
    '''
    o = np.empty(P.shape[0]) # initialize the evaluation vector
    for i in range(P.shape[0]):
        o[i] = q(P[i]) # fill the evaluation vector with the objective function values for each individual in the population
    return o

def mutation(P, sigma):
    '''
    Operation of generating a point from the vicinity of the mutated point
    :param P: Population
    :param sigma: Mutation strength
    :return M: Returns the mutated population
    '''
    M = np.copy(P) # initiate the mutated population
    for i in range(P.shape[0]):
        # for each gene (j) from individual i
        for j in range(P.shape[1]):
            # add a random value to the gene from a normal distribution scaled by sigma parameter
            M[i, j] = P[i, j] + sigma * np.random.normal(0,1)
            
            # cubic limitation -100, 100
            if (M[i, j] > 100):
                M[i,j] = 100
            elif (M[i,j] < -100):
                M[i,j] = -100
    return M

def tournament_selection(P, o, mi, S):
    '''
    Tournament selection with 2-individual tournaments - first, we select a group of individuals, then 
    the best from this group is chosen. The probability of an individual's reproduction depends on its rank,
    but also on whether the individual is selected for the tournament
    :param P: population
    :param o: evaluation of population vector
    :param mi: number of individuals
    :param S: tournament size
    :return R: returns the population remaining after selection
    '''
    R = np.empty((mi, P.shape[1]))
    
    for j in range (mi):
        tournament_group = np.empty((S, P.shape[1]+1))
        
        for i in range(S):
            # Choose a random individual number
            random_individual = np.random.choice(mi)
            # Add the randomly selected individual to the tournament group along with its evaluation
            tournament_group[i] = np.hstack((o[random_individual], P[random_individual]))
            
        num_winner = np.argmin(tournament_group[:, 0]) # Take the winner's number as the individual with the best evaluation
        tournament_winner = tournament_group[num_winner] # Take the winning individual
        R[j] = tournament_winner[1:] # Insert the winning individual into the selected population set
    
    return R

def generational_succession(P, M, o, o_m):
    '''
    Generational succession operation - decides which individuals will survive to the next generation
    In the case of generational succession, the population of mutants proceeds further
    '''
    return M, o_m

def evolutionary_algorithm(qx, P, mi, sigma, t_max):
    '''
    Evolutionary algorithm with tournament selection and generational succession, without crossover
    :param qx: evaluation function
    :param P_0: initial population
    :param mi: number of individuals
    :param sigma: mutation strength
    :param t_max: number of iterations
    :return x_best, o_best: returns the best found solution and its evaluation
    ''' 
    t = 0
    o = evaluate(qx, P)
    x_best, o_best = find_best(P, o)
    while not stop(t, t_max): 
        R = tournament_selection(P, o, mi, 2) # R - temporary population
        M = mutation(R, sigma) # M - mutant population
        o_m = evaluate(qx, M) # evaluation of the mutant population
        x_star_t, o_star_t = find_best(M, o_m) # search for the best individual from the mutant population
        
        # update the best individual if a better one is found
        if(o_star_t <= o_best):
            o_best = o_star_t
            x_best = x_star_t
        
        P, o = generational_succession(P, M, o, o_m)
        t = t+1
    
    return x_best, o_best
