from english_version import *

def EA_results_generator(qlist, mi_min, mi_max, mi_step, sigma_min, sigma_max, sigma_step, UPPER_BOUND, DIMENSIONALITY, budget):
    '''
    Testing method for the evolutionary algorithm, printing results in an .xlsx file
    :param qlist: list of functions to test
    :param mi_min: minimum mi parameter
    :param mi_max: maximum mi parameter
    :param mi_step: mi parameter step change
    :param sigma_min: minimum sigma parameter
    :param sigma_max: maximum sigma parameter
    :param sigma_step: sigma parameter step change
    :param UPPER_BOUND: cubic space limitation -UPPER_BOUND, UPPERBOUND
    :param DIMENSIONALITY: number of dimensions in which we optimize the given function
    :param budget: budget (number of evaluations)
    '''
    x_o_results_q2 = []
    x_o_results_q13 = []
    for q in qlist:
        for i in np.arange(sigma_min, sigma_max + sigma_step, sigma_step):
            for j in range(mi_min, mi_max + mi_step, mi_step):
                sigma = i
                mi = j  # number of individuals in the initial population
                P_0 = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=(mi, DIMENSIONALITY))
                t_max = budget / mi
                o_values = []  # initialization of the obtained evaluations list
                
                # demonstration of calculating results for one set of parameters
                print(".", end="", flush=True)
                
                x_best, o_best = evolutionary_algorithm(q, P_0, mi, sigma, t_max)          
                if q == q2:
                    x_o_results_q2.append({
                        "σ": f"{sigma:.2f}".replace(".", ","),
                        "μ": mi,
                        "o*": f"{o_best:.2f}".replace(".", ","),
                        "x*": [f"{val:3.2f}".replace(".", ",") for val in x_best],
                    })
                else:
                    x_o_results_q13.append({
                        "σ": f"{sigma:.2f}".replace(".", ","),
                        "μ": mi,
                        "o*": f"{o_best:.2f}".replace(".", ","),
                        "x*": [f"{val:3.2f}".replace(".", ",") for val in x_best],
                    })
    
    x_o_results_q2_df = pd.DataFrame(x_o_results_q2)
    x_o_results_q13_df = pd.DataFrame(x_o_results_q13)
    
    # sort by evaluation value
    x_o_results_q2_df = x_o_results_q2_df.sort_values(by="o*")
    x_o_results_q13_df = x_o_results_q13_df.sort_values(by="o*")
    
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)
    
    # save results in an .xlsx file
    x_o_results_q2_df.to_excel("x_o_results_q2.xlsx", index=False)
    x_o_results_q13_df.to_excel("x_o_results_q13.xlsx", index=False)
    
    # optionally print results in console
    # print("\nFunction: f2 CEC 2017")
    # print(x_o_results_q2_df)
    # print("Function: f13 CEC 2017")
    # print(x_o_results_q13_df)

def EA_param_test(qlist, mi_min, mi_max, mi_step, sigma_min, sigma_max, sigma_step, UPPER_BOUND, DIMENSIONALITY, budget, num_test):
    '''
    Testing method for EA, printing the statistics of multiple EA runs in an .xlsx file
    :param qlist: list of functions to be tested
    :param mi_min: minimum value of mi parameter
    :param mi_max: maximum value of mi parameter
    :param mi_step: step change of mi parameter
    :param sigma_min: minimum value of sigma parameter
    :param sigma_max: maximum value of sigma parameter
    :param sigma_step: step change of sigma parameter
    :param UPPER_BOUND: cubic space limitation -UPPER_BOUND, UPPERBOUND
    :param DIMENSIONALITY: number of dimensions in which we optimize the given function
    :param budget: evaluation budget (number of evaluations)
    :param num_test: number of runs of the algorithm for each set of tested parameters
    '''
    for q in qlist:
        results = []  # results for the currently tested function
        for i in np.arange(sigma_min, sigma_max + sigma_step, sigma_step):
            for j in range(mi_min, mi_max + mi_step, mi_step):
                sigma = i
                mi = j  # number of individuals in the initial population
                o_values = []  # initialization of the obtained evaluations list
                
                # demonstration of calculating results for one group of parameters
                print(".", end="", flush=True)
                    
                for k in range(1, num_test + 1):
                    # randomly generate a population of a given size and gene value range
                    P_0 = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=(mi, DIMENSIONALITY))
                    t_max = budget / mi
                    
                    # call the evolutionary algorithm to find the best solutions and save them on the list
                    x_best, o_best = evolutionary_algorithm(q, P_0, mi, sigma, t_max)
                    o_values.append(o_best)
                    
                # calculate values for the table
                min_o = np.min(o_values)
                mean_o = np.mean(o_values)
                max_o = np.max(o_values)
                std_o = np.std(o_values)
                
                # insert values into the table
                results.append({
                    "σ": f"{sigma:.2f}".replace(".", ","),
                    "μ": mi,
                    "min": f"{min_o:.2f}".replace(".", ","),
                    "avg": f"{mean_o:.2f}".replace(".", ","),
                    "max": f"{max_o:.2f}".replace(".", ","),
                    "std": f"{std_o:.2f}".replace(".", ",")
                })

        # save results in an .xlsx file
        results_df = pd.DataFrame(results)
        q_name = 'q2' if q == q2 else 'q13'
        results_df.to_excel(f"results_{q_name}.xlsx", index=False)
    
    return True

def EA_budget_test(qx, mi, sigma, UPPER_BOUND, DIMENSIONALITY, budget1, budget2, num_test):
    '''
    Testing method for the budget size in EA, printing results in an .xlsx file
    :param qx: objective function
    :param mi: value of mi
    :param sigma: value of sigma
    :param UPPER_BOUND: cubic space limitation -UPPER_BOUND, UPPERBOUND
    :param DIMENSIONALITY: number of dimensions in which we optimize the given function
    :param budget1, budget2: budgets used in tests (number of evaluations)
    :param num_test: number of runs of the algorithm for each set of tested parameters
    '''
    results = []  # results for the currently tested function
    
    for budget in [budget1, budget2]:
        o_values = []  # initialization of the obtained evaluations list
        for k in range(1, num_test + 1):
            # randomly generate a population of a given size and gene value range
            P_0 = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=(mi, DIMENSIONALITY))
            
            t_max = budget / mi    
            x_best, o_best = evolutionary_algorithm(qx, P_0, mi, sigma, t_max)
            o_values.append(o_best)
            
        # calculate values for the table
        min_o = np.min(o_values)
        mean_o = np.mean(o_values)
        max_o = np.max(o_values)
        std_o = np.std(o_values)
        
        # insert values into the table
        results.append({
            "budget": budget,
            "σ": f"{sigma:.2f}".replace(".", ","),
            "μ": mi,
            "min": f"{min_o:.2f}".replace(".", ","),
            "avg": f"{mean_o:.2f}".replace(".", ","),
            "max": f"{max_o:.2f}".replace(".", ","),
            "std": f"{std_o:.2f}".replace(".", ",")
        })

        # save results in an .xlsx file
        results_df = pd.DataFrame(results)
        q_name = 'q2' if qx == q2 else 'q13'
        results_df.to_excel(f"budget_test_results_{q_name}.xlsx", index=False)
    
    return True

def main():
    budget = 10000  # available evaluation budget
    UPPER_BOUND = 100
    DIMENSIONALITY = 10
    # selection of mutation strength value range (changes every sigma_step)
    sigma_min = 1.50
    sigma_max = 1.55
    sigma_step = 0.05
    # selection of the range of the number of individuals in the population
    mi_min = 14
    mi_max = 16
    mi_step = 2
    qlist = [q2, q13]  # list of functions to be tested
    num_test = 25  # number of algorithm runs for each set of parameters

    # EA_param_test and EA_budget_test functions calls can be uncommented for testing
    # Uncomment the functions below for actual execution

    # EA_param_test(qlist, mi_min, mi_max, mi_step, sigma_min, sigma_max, sigma_step, UPPER_BOUND, DIMENSIONALITY, budget, num_test)
    
    # EA_budget_test example usage
    # EA_budget_test(qx, mi, sigma, UPPER_BOUND, DIMENSIONALITY, budget1, budget2, num_test)
    
    # Example call for generating results
    # EA_results_generator(qlist, mi_min, mi_max, mi_step, sigma_min, sigma_max, sigma_step, UPPER_BOUND, DIMENSIONALITY, budget)

if __name__ == "__main__":
    main()
