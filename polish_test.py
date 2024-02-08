from polish_version import *

def AE_results_generator(qlist, mi_min, mi_max, mi_step, sigma_min, sigma_max, sigma_step, UPPER_BOUND, DIMENSIONALITY, budget):
    '''
    Metoda testująca algorytm ewolucji wypisująca wyniki w pliku .xlsx
    :param qlist: lista funkcji do przetestowania
    :param mi_min: minimalny parametr mi
    :param mi_max: maksymalny parametr mi
    :param mi_step: krok zmian parametru mi
    :param sigma_min: minimalny parametr sigma
    :param sigma_max: maksymalny parametr sigma
    :param sigma_step: krok zmian parametru sigma
    :param UPPER_BOUND: ograniczenie kostkowe przestrzeni -UPPER_BOUND, UPPERBOUND
    :param DIMENSIONALITY: liczba wymiarów w których optymalizujemy daną funkcję
    :param budget: budżet (liczba ewaluacji)
    '''
    x_o_results_q2 = []
    x_o_results_q13 = []
    for q in qlist:
        for i in np.arange(sigma_min, sigma_max + sigma_step, sigma_step):
            for j in range (mi_min, mi_max + mi_step, mi_step):
                sigma = i
                mi = j # liczba osobników w populacji początkowej
                P_0 = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=(mi, DIMENSIONALITY))
                t_max = budget / mi
                o_values = [] # inicjalizacja listy uzyskanych ocen
                
                # demonstracja wyliczenia wyników dla jednej grupy parametrów
                print(".", end="", flush=True)
                
                x_best, o_best = evolutionary_algorithm(q, P_0, mi, sigma, t_max)          
                if(q == q2):
                    x_o_results_q2.append({
                        "\u03C3": f"{sigma:.2f}".replace(".", ","),
                        "\u03BC": mi,
                        "o*": f"{o_best:.2f}".replace(".", ","),
                        "x*": [f"{val:3.2f}".replace(".", ",") for val in x_best],
                    })
                else:
                    x_o_results_q13.append({
                        "\u03C3": f"{sigma:.2f}".replace(".", ","),
                        "\u03BC": mi,
                        "o*": f"{o_best:.2f}".replace(".", ","),
                        "x*": [f"{val:3.2f}".replace(".", ",") for val in x_best],
                    })
    
    x_o_results_q2_df = pd.DataFrame(x_o_results_q2)
    x_o_results_q13_df = pd.DataFrame(x_o_results_q13)
    
    # posortuj według wartości oceny
    x_o_results_q2_df = x_o_results_q2_df.sort_values(by="o*")
    x_o_results_q13_df = x_o_results_q13_df.sort_values(by="o*")
    
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)
    
    # zapisz wyniki w pliku .xlsx
    x_o_results_q2_df.to_excel("x_o_results_q2.xlsx", index=False)
    x_o_results_q13_df.to_excel("x_o_results_q13.xlsx", index=False)
    
    # opcjonalne wypisanie wyników w konsoli
    # print("\nFunkcja: f2 CEC 2017")
    # print(x_o_results_q2_df)
    # print("Funkcja: f13 CEC 2017")
    # print(x_o_results_q13_df)

def AE_param_test(qlist, mi_min,  mi_max, mi_step, sigma_min, sigma_max, sigma_step, UPPER_BOUND, DIMENSIONALITY, budget, num_test):
    '''
    Metoda testująca AE wypisująca statystyki wyników wielu uruchomień AE w pliku .xlsx
    :param qlist: lista funkcji do przetestowania
    :param mi_min: minimalny parametr mi
    :param mi_max: maksymalny parametr mi
    :param mi_step: krok zmian parametru mi
    :param sigma_min: minimalny parametr sigma
    :param sigma_max: maksymalny parametr sigma
    :param sigma_step: krok zmian parametru sigma
    :param UPPER_BOUND: ograniczenie kostkowe przestrzeni -UPPER_BOUND, UPPERBOUND
    :param DIMENSIONALITY: liczba wymiarów w których optymalizujemy daną funkcję
    :param budget: budżet (liczba ewaluacji)
    :param num_test: liczba uruchomień algorytmu dla każdego z testowanych zbiorów parametrów
    '''
    for q in qlist:
        results = []  # wyniki dla aktualnie badanej funkcji
        for i in np.arange(sigma_min, sigma_max + sigma_step, sigma_step):
            for j in range (mi_min, mi_max + mi_step, mi_step):
                sigma = i
                mi = j # liczba osobników w populacji początkowej
                o_values = [] # inicjalizacja listy uzyskanych ocen
                
                # demonstracja wyliczenia wyników dla jednej grupy parametrów
                print(".", end="", flush=True)
                    
                for k in range(1, num_test + 1):
                    # wylosuj losową populację o danej wielkości i danym zakresie wartości genów
                    P_0 = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=(mi, DIMENSIONALITY))
                    t_max = budget / mi
                    
                    # wywołaj algorytm ewolucyjny do znalezienia najlepszych rozwiązań i zapisz je na liście
                    x_best, o_best = evolutionary_algorithm(q, P_0, mi, sigma, t_max)
                    o_values.append(o_best)
                    
                # wylicz wartości do tabeli
                min_o = np.min(o_values)
                mean_o = np.mean(o_values)
                max_o = np.max(o_values)
                std_o = np.std(o_values)
                
                # wstaw wartości do tabeli
                results.append({
                    "\u03C3": f"{sigma:.2f}".replace(".", ","),
                    "\u03BC": mi,
                    "min": f"{min_o:.2f}".replace(".", ","),
                    "śr": f"{mean_o:.2f}".replace(".", ","),
                    "max": f"{max_o:.2f}".replace(".", ","),
                    "std": f"{std_o:.2f}".replace(".", ",")
                })

        # zapisz wyniki w pliku .xlsx
        results_df = pd.DataFrame(results)
        q_name = 'q2' if q == q2 else 'q13'
        results_df.to_excel(f"results_{q_name}.xlsx", index=False)
    
    return True

def AE_budget_test(qx, mi, sigma, UPPER_BOUND, DIMENSIONALITY, budget1, budget2, num_test):
    '''
    Metoda testująca wielkość budżetu w AE wypisująca wyniki w pliku .xlsx
    :param qx: funkcja celu
    :param mi: wartość mi
    :param sigma: wartość sigma
    :param UPPER_BOUND: ograniczenie kostkowe przestrzeni -UPPER_BOUND, UPPERBOUND
    :param DIMENSIONALITY: liczba wymiarów w których optymalizujemy daną funkcję
    :param budget1 budget2: budżety wykrozystywane w testach (liczba ewaluacji)
    :param num_test: liczba uruchomień algorytmu dla każdego z testowanych zbiorów parametrów
    '''
    results = []  # wyniki dla aktualnie badanej funkcji
    
    for budget in [budget1, budget2]:
        o_values = [] # inicjalizacja listy uzyskanych ocen
        for k in range(1, num_test + 1):
            # wylosuj losową populację o danej wielkości i danym zakresie wartości genów
            P_0 = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=(mi, DIMENSIONALITY))
            
            t_max = budget / mi    
            x_best, o_best = evolutionary_algorithm(qx, P_0, mi, sigma, t_max)
            o_values.append(o_best)
            
        # wylicz wartości do tabeli
        min_o = np.min(o_values)
        mean_o = np.mean(o_values)
        max_o = np.max(o_values)
        std_o = np.std(o_values)
        
        # wstaw wartości do tabeli
        results.append({
            "budżet": budget,
            "\u03C3": f"{sigma:.2f}".replace(".", ","),
            "\u03BC": mi,
            "min": f"{min_o:.2f}".replace(".", ","),
            "śr": f"{mean_o:.2f}".replace(".", ","),
            "max": f"{max_o:.2f}".replace(".", ","),
            "std": f"{std_o:.2f}".replace(".", ",")
        })

        # zapisz wyniki w pliku .xlsx
        results_df = pd.DataFrame(results)
        q_name = 'q2' if qx == q2 else 'q13'
        results_df.to_excel(f"budget_test_results_{q_name}.xlsx", index=False)
    
    return True

def main():
    budget = 10000 # budżet dostępnych ewaluacji funkcji celu
    UPPER_BOUND = 100
    DIMENSIONALITY = 10
    # wybór zakresu wartości siły mutacji (zmiany co sigma_step)
    sigma_min = 1.50
    sigma_max = 1.55
    sigma_step = 0.05
    # wybór zakresu liczby osobników w populacji
    mi_min = 14
    mi_max = 16
    mi_step = 2
    qlist = [q2, q13] # lista testowanych funkcji
    num_test = 25 # liczba uruchomień algorytmu dla każdego z parametrów
    #AE_param_test(qlist, mi_min,  mi_max, mi_step, sigma_min, sigma_max, sigma_step, UPPER_BOUND, DIMENSIONALITY, budget, num_test)
    
    budget1 = 10000
    budget2 = budget1 * 5
    qx = q2
    mi = 14
    sigma = 1.45
    #AE_budget_test(qx, mi, sigma, UPPER_BOUND, DIMENSIONALITY, budget1, budget2, num_test)
    
    qx = q13
    mi = 138
    sigma = 0.35
    #AE_budget_test(qx, mi, sigma, UPPER_BOUND, DIMENSIONALITY, budget1, budget2, num_test)
    
    AE_results_generator(qlist, mi_min, mi_max, mi_step, sigma_min, sigma_max, sigma_step, UPPER_BOUND, DIMENSIONALITY, budget)

if __name__ == "__main__":
    main()