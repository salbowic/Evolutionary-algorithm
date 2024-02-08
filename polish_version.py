import numpy as np
import pandas as pd
import cec2017

from cec2017.functions import f2
from cec2017.functions import f13
from colorama import Fore

#funkcja 2 z CEC 2017
def q2(x):
    return f2(x)

#funkcja 13 z CEC 2017
def q13(x):
    return f13(x)


def find_best(P, o):
    '''
    Metoda, która znajduje najlepszego osobnika (minimalizacja - najmniejszy wynik funkcji celu)
    :param P: populacja
    :param o: ocena populacji
    :return x_best, o_best: Zwraca najlepsze rozwiązanie i jego ocenę z danej populacji
    '''
    # Weź numer argumentu najlepszej oceny z macierzy ocen
    o_best_arg = np.argmin(o)
    # Ustaw najlepszą ocenę i najlepszy punkt
    o_best = o[o_best_arg]
    x_best = P[o_best_arg]
    return x_best, o_best


def stop(t,t_max):
    '''
    Kryterium stopu
    :param t: numer iteracji
    :param t_max: liczba iteracji 
    '''
    if t > t_max:
        return True


def evaluate(q, P):
    '''
    Operacja ewaluacji funkcji na danej populacji
    :param q: funkcja oceny
    :param P: Populacja
    :return o: wektor ocen populacji
    '''
    o = np.empty(P.shape[0]) # inicjalizacja wektora ocen
    for i in range(P.shape[0]):
        o[i] = q(P[i]) # uzupełnienie wektora ocen o wartości funkcji celu dla każdego osobnika z populacji
    return o


def mutation(P, sigma):
    '''
    Operacja wygenerowania punktu z otoczenia punktu mutowanego
    :param P: Populacja
    :param sigma: Siła mutacji
    :return M: Zwraca populację zmutowaną
    '''
    M = np.copy(P) # zainicjuj zmutowaną populację
    for i in range(P.shape[0]):
        # dla każdego genu (j) z osobnika i
        for j in range(P.shape[1]):
            # dodaj losową wartość do genu z rozkładu normalnego przeskalowaną przez parametr sigma
            M[i, j] = P[i, j] + sigma * np.random.normal(0,1)
            
            # ograniczenie kostkowe -100, 100
            if (M[i, j] > 100):
                M[i,j] = 100
            elif (M[i,j] < -100):
                M[i,j] = -100
    return M


def tournament_selection(P, o, mi, S):
    '''
    Selekcja turniejowa z turniejami 2 osobnikowymi - najpierw wybieramy grupę osobników, następnie 
    najlepszy z tej grupy jest wybierany. Prawdopodobieństwo reprodukcji osobnika zależy od jego rangi,
    ale zależy również od tego, czy osobnik zostanie wybrany do turnieju
    :param P: populacja
    :param o: ocena populacji
    :param mi: liczba osobników
    :param S: rozmiar turnieju
    :return R: zwraca populacje pozostałą po selekcji
    '''
    R = np.empty((mi, P.shape[1]))
    
    for j in range (mi):
        tournament_group = np.empty((S, P.shape[1]+1))
        
        for i in range(S):
            # Wybierz losowy numer osobnika
            random_individual = np.random.choice(mi)
            # Dodaj do grupy turniejowej osobnika o wybranym losowo numerze wraz z jego oceną
            tournament_group[i] = np.hstack((o[random_individual], P[random_individual]))
            
        num_winner = np.argmin(tournament_group[:, 0]) # Weź nr wygranego jako nr osobnika z najlepszą oceną
        tournament_winner = tournament_group[num_winner] # Weź wygranego osobnika
        R[j] = tournament_winner[1:] # Wstaw wygranego osobnika do zbioru wyselekcjonowanej populacji
    
    return R


def generational_succession(P, M, o, o_m):
    '''
    Operacja sukcesji generacyjnej - decyduje, które osobniki przeżyją do następnej generacji
    W przypadku sukcesji generacyjnej, dalej przechodzi populacja mutantów
    '''
    return M, o_m


def evolutionary_algorithm(qx, P, mi, sigma, t_max):
    '''
    Algorytm ewolucyjny z selekcją turniejową i sukcesją generacyjną, bez krzyżowania
    :param qx: funkcja oceny
    :param P_0: populacja początkowa
    :param mi: liczba osobników
    :param sigma: siła mutacji
    :param t_max: liczba iteracji
    :return x_best, o_best: zwraca najlepsze znalezione rozwiązanie i jego ocenę
    ''' 
    t = 0
    o = evaluate(qx, P)
    x_best, o_best = find_best(P, o)
    while not stop(t,t_max): 
        R = tournament_selection(P, o, mi, 2) # R - populacja tymczasowa
        M = mutation(R,sigma) # M - populacja mutantów
        o_m = evaluate(qx, M) # ocena populacji mutantów
        x_star_t, o_star_t = find_best(M, o_m) # szukanie najlepszego osobnika z populacji mutantów
        
        # aktualizacja najlepszego osobnika, jeżeli znaleziono lepszego
        if(o_star_t <= o_best):
            o_best = o_star_t
            x_best = x_star_t
        
        P, o = generational_succession(P, M, o, o_m)
        t = t+1
    
    return x_best, o_best


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