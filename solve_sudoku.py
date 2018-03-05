"""Solve a given Sudoku problem using Genetic Algorithms.

This program uses an Elitist Generational Genetic Algorithm to solve a
specified Sudoku problem contained within a csv file. The best solution found
when the stopping criteria has been reached will be displayed on screen along
with the fitness score. It will then be saved as a csv file in the
solved_sudoku folder.

Flags:
    -i, --input         Name of the grid file in csv_sudoku to solve.
    -p, --population    Population size.
Example:
    $ python convert_grid.py -i Grid1.csv -p 10 -e -m -d

"""

from sudoku_utils import read_sudoku, print_sudoku
from os.path import basename, join
from random import randint, random, sample, randrange, shuffle
import numpy as np
import argparse

CSV_PATH = "./csv_sudoku"
OUT_PATH = "./solved_sudoku"


def get_rows(sudoku_gene):
    """Return a list representing the rows.

    This function converts a 1 dimensional list which represents the sudoku
    gene and converts it into a list of the columns represented as lists.

    Args:
        sudoku_gene (list)  List representing the sudoku grid.
    Returns:
        A two dimensional list of sudoku columns.

    """
    row_array = []
    for _ in range(9):
        row_array.append([])
    for i, number in enumerate(sudoku_gene):
        row_array[int(i/9)].append(number)
    return row_array


def get_columns(sudoku_gene):
    """Return a list representing the columns.

    This function converts a 1 dimensional list which represents the sudoku
    gene and converts it into a list of the columns represented as lists.

    Args:
        sudoku_gene (list)  List representing the sudoku grid.
    Returns:
        A two dimensional list of sudoku columns.

    """
    column_array = []
    for _ in range(9):
        column_array.append([])
    for i, number in enumerate(sudoku_gene):
        column_array[i % 9].append(number)
    return column_array


def get_boxes(sudoku_gene):
    """Return a list representing the 3x3 boxes.

    This function converts a 1 dimensional list which represents the sudoku
    gene and converts it into a list of the 3x3 boxes represented as lists.

    Args:
        sudoku_gene (list)  List representing the sudoku grid.
    Returns:
        A two dimensional list of sudoku boxes.

    """
    box_array = [[]]
    for i in range(3):
        box_array.append([])
        for _ in range(3):
            box_array[i].append([])
    for i, number in enumerate(sudoku_gene):
        box_array[int(i/9/3)][int((i % 9)/3)].append(number)
    return sum(box_array, [])


def check_positions(to_check, original):
    """Check that the preset grid numbers are still present in the original.

    Each number within a gene is checked against the original fixed gene array,
    if the fixed non zero number differes from that in the gene to be checked
    then the gene to be checked is not valid.

    Args:
        to_check (list)  List representing the sudoku grid to check.
        orignal (list)  List representing the fixed sudoku grid.
    Returns:
        True if the list is valid, false if not

    """
    correct = True
    for i in range(len(original)):
        if original[i] != 0 and original[i] != to_check[i]:
            print("Incorrect index:", i)
            correct = False
    return correct


def check_valid(to_check, original):
    """Check that the preset grid numbers form a valid solution.

    Each number within a gene is checked against the original fixed gene array,
    while each row, column, and grid is checked to ensure that it contains
    a valid permutation of the set 1 to 9.

    Args:
        to_check (list)  List representing the sudoku grid to check.
        orignal (list)  List representing the fixed sudoku grid.
    Returns:
        True if the list is valid, False if not.

    """
    correct = check_positions(to_check, original)
    if correct:
        for row in get_rows(to_check):
            correct = len(set(row)) == 9
            correct = sum(row) == 45
    if correct:
        for column in get_columns(to_check):
            correct = len(set(column)) == 9
            correct = sum(column) == 45
    if correct:
        for box in get_boxes(to_check):
            correct = len(set(box)) == 9
            correct = sum(box) == 45

    return correct


def initalise_population(sudoku_gene, pop_size):
    """Initalise a population of Sudoku solutions.

    This function creates an array of randomly generated Sudoku solutions
    based off of the given sudoku grid. Only blank indexes have a random
    integer in the range of 0 and 9 generated. In this way not all, if any
    of the solutions generated will be valid given the rules of Sudoku.

    Args:
        sudoku_gene (list)      List representing the sudoku grid.
        fixed_indicies (List)   Indexes which are not to be changed.
        pop_size (int)          Number of individuals in the population.

    Returns:
        A list of randomly generated Sudoku solutions.

    """
    population_array = np.zeros((pop_size, len(sudoku_gene)), dtype=np.uint8)
    for i in range(pop_size):
        new_pop = get_rows(sudoku_gene)
        for j, box in enumerate(new_pop):
            # Calculate the complete set
            rand_set = [i for i in range(len(box)+1)]
            for n in set(box):
                # Remove existing numbers from the set
                rand_set.remove(n)
            for k, num in enumerate(box):
                # For unset space, set the space to a number
                # from the set then remove that number.
                if num == 0:
                    rand_int = sample(rand_set, 1)[0]
                    new_pop[j][k] = rand_int
                    rand_set.remove(rand_int)

        population_array[i] = sum(new_pop, [])
    return population_array


def evaluate_individual(sudoku_gene):
    """Get the fitness score of an individual within a population.

    This function calculates the fitness score of an indivudal for the length
    of the set of numbers for each box and column within a solution.

    Args:
        sudoku_gene (list)      List representing the sudoku grid.
    Returns:
        A float value between 0 and 1 with 1 being the best and 0 the worst
        score.

    """
    score = 0

    for box in get_boxes(sudoku_gene):
        score += len(set(box))
    for column in get_columns(sudoku_gene):
        score += len(set(column))

    return score/162


def evaluate_population(population_array):
    """Evaluate the fitness score of each individual in a population.

    For each member in a population, calculate their individual fitness score
    and append it to a list to create an array of all the fitnesses in the
    population.

    Args:
        population_array (list) A list containing all the solutions in a
                                population.
    Returns:
        An array of all the fitness scores in a population.

    """
    fitness_array = np.zeros((population_array.shape[0]))
    for i, individual in enumerate(population_array):
        fitness_array[i] = evaluate_individual(individual)
    return fitness_array


def duel_tournament_selection(population_array, fitness_array, t_select,
                              t_size, duel_selection=0.8):
    """Select parents from the population using tournament selection.

    Generate a tournament pool and select the best individual from it to be
    a parent for the next generation. However, this varient of the tournament
    selector will select the worst individual at a rate goverend by the duel
    selection chance.

    Args:
        population_array (list) A list containing all the solutions in a
                                population.
        fitness_array (list)    A list containing all the fitnesses scores of a
                                population.
        t_select (int)          The number of parents to select.
        t_size (int)            The number of individuals within each
                                tournament.
        duel_selector (float)   The chance to select the best canditate, worse
                                otherwise (default 0.8).
    Returns:
        An array of the genes of the parents selected

    """
    parents = np.zeros((t_select, population_array.shape[1]), dtype=np.uint32)

    for i in range(t_select):
        tournament_list = sample([j for j in range(len(population_array))],
                                 t_size)
        # Shuffle the list to prevent argmax from selecting
        # only one gene if there are multiple different genes
        # with the best fitness functions.
        shuffle(tournament_list)
        t_fitnesses = [fitness_array[j] for j in tournament_list]

        if random() < duel_selection:
            winner_index = tournament_list[np.argmax(t_fitnesses)]
        else:
            winner_index = tournament_list[np.argmin(t_fitnesses)]
        parents[i] = population_array[winner_index]

    return parents


def tournament_selection(population_array, fitness_array, t_select, t_size):
    """Select parents from the population using tournament selection.

    Generate a tournament pool and select the best individual from it to be
    a parent for the next generation. This operator will always select the
    best individual from the tournament.

    Args:
        population_array (list) A list containing all the solutions in a
                                population.
        fitness_array (list)    A list containing all the fitnesses scores of a
                                population.
        t_select (int)          The number of parents to select.
        t_size (int)            The number of individuals within each
                                tournament.
        duel_selector (float)   The chance to select the best canditate, worse
                                otherwise
    Returns:
        An array of the genes of the parents selected.

    """
    parents = np.zeros((t_select, population_array.shape[1]), dtype=np.uint32)

    for i in range(t_select):
        tournament_list = sample([j for j in range(len(population_array))],
                                 t_size)
        # Shuffle the list to prevent argmax from selecting
        # only one gene if there are multiple different genes
        # with the best fitness functions.
        shuffle(tournament_list)
        t_fitnesses = [fitness_array[j] for j in tournament_list]
        winner_index = tournament_list[np.argmax(t_fitnesses)]
        parents[i] = population_array[winner_index]

    return parents


def row_crossover(parent_1, parent_2, crossover_rate=1):
    """Create two offspring genes from two parent genes.

    This crossover operator shuffles the rows for the two parents to create
    two children. Both children contain a combination of the rows of one
    parent and the rows of the other, randomly determined.

    Args:
        parent_1 (list) A numerical list representing the genes of a parent.
        parent_2 (list) A numerical list representing the genes of a parent.
        crossover_rate (float) Chance that a crossover will occur (default 1).

    Returns:
        A tuple containing the pair of children.

    """
    if random() < crossover_rate:
        offspring1 = []
        offspring2 = []
        binary_mask = [randint(0, 1) for _ in range(0, len(parent_1), 9)]
        for i, binary in enumerate(binary_mask):
            start = i * 9
            end = i * 9 + 9
            for j in range(start, end):
                    if binary:
                        offspring1.append(parent_1[j])
                        offspring2.append(parent_2[j])
                    else:
                        offspring1.append(parent_2[j])
                        offspring2.append(parent_1[j])
    else:
        offspring1 = parent_1.copy()
        offspring2 = parent_2.copy()
    return offspring1, offspring2


def create_children(parents, n_children, crossover_rate=1):
    """Create a number of children from a list of parents.

    This function selects two parents from the list of parents and creates
    two kids from the function.

    Args:
        parents (list) A numerical list containting the genes of the parents.
        n_children (int) The number of children to create, must be even.
        crossover_rate (float) Chance that a crossover will occur (default 1).
    Returns:
        A list of the genes of the new children.

    """
    offspring = []
    for i in range(int(n_children/2)):
        parent_sample = sample(list(parents), 2)
        parent_1_gene = parent_sample[0]
        parent_2_gene = parent_sample[1]
        child_1, child_2 = row_crossover(parent_1_gene, parent_2_gene,
                                         crossover_rate)
        offspring.append(list(child_1))
        offspring.append(list(child_2))
    return np.asarray(offspring)


def swap_mutate_offspring(offspring_array, mutation_rate, fixed_indicies):
    """Mutate all the offspring to produce distrinct offspring.

    This genetic operator selects a row at random for an offspring and two
    points within that row. If the two points are distinct and are not fixed
    then the two points will be swapped. This is repeated for each child in the
    offspring array based on the mutation rate.

    Args:
        offspring_array (list) A numerical list of the genes of all the
                                children to be mutated.
        mutation_rate (float) The chance that a mutation will occur.
        fixed_indicies (list) A list of the indicies which are not to be
                              changed by the mutation.
    Returns:
        A list of the genes of the new children.

    """
    for i, child in enumerate(offspring_array):
        if random() < mutation_rate:
            mutant = child.copy()
            # Generate the swap indexes.
            rand_row = randrange(0, len(mutant), 9)
            swap_index_1 = rand_row + randint(0, 8)
            swap_index_2 = rand_row + randint(0, 8)
            # Ensure that the indicies are not fixed or equal.
            while (swap_index_1 == swap_index_2 or fixed_indicies[swap_index_1]
                   or fixed_indicies[swap_index_2]):
                swap_index_1 = rand_row + randint(0, 8)
                swap_index_2 = rand_row + randint(0, 8)
            # Swap the contents of the indicies.
            swap_temp = mutant[swap_index_1]
            mutant[swap_index_1] = mutant[swap_index_2]
            mutant[swap_index_2] = swap_temp
            offspring_array[i] = mutant
    return offspring_array


def multi_mutate_offspring(offspring_array, mutation_rate, fixed_indicies,
                           n_mutate=5):
    """Mutate all the offspring to produce distrinct offspring.

    This genetic operator selects a row at random for an offspring and two
    points within that row. If the two points are distinct and are not fixed
    then the two points will be swapped. This is repeated for each child in the
    offspring array based on the mutation rate. This mutation can occur
    multiple times to the same individual.

    Args:
        offspring_array (list) A numerical list of the genes of all the
                                children to be mutated.
        mutation_rate (float) The chance that a mutation will occur.
        fixed_indicies (list) A list of the indicies which are not to be
                              changed by the mutation.
        n_mutate (int) The number of times the mutation is applied to a
                        child (default 5).
    Returns:
        A list of the genes of the new children.

    """
    for i, child in enumerate(offspring_array):
        if random() < mutation_rate:
            for n in range(randint(1, n_mutate)):
                mutant = child.copy()
                # Generate the swap indexes.
                rand_row = randrange(0, len(mutant), 9)
                swap_index_1 = rand_row + randint(0, 8)
                swap_index_2 = rand_row + randint(0, 8)
                # Ensure that the indicies are not fixed or equal.
                while (swap_index_1 == swap_index_2 or
                       fixed_indicies[swap_index_1]
                       or fixed_indicies[swap_index_2]):
                    swap_index_1 = rand_row + randint(0, 8)
                    swap_index_2 = rand_row + randint(0, 8)
                # Swap the contents of the indicies.
                swap_temp = mutant[swap_index_1]
                mutant[swap_index_1] = mutant[swap_index_2]
                mutant[swap_index_2] = swap_temp
                offspring_array[i] = mutant
    return offspring_array


def get_elites(population, fitness_array, n_elites):
    """Get the the best individuals from a population.

    Find the top individuals within a population and then save them to an
    array.

    Args:
        population (list) A list of the genes for every indivudal in a
        population.
        fitness_array (list) A list containing all the fitnesses scores of a
                              population.
        n_elites (int) The number of elites to select from a population.
    Returns:
        A list of the top individuals of the population.

    """
    elites = np.zeros((n_elites, population.shape[1]), dtype=np.uint8)
    zipped = list(sorted(zip(population, fitness_array), key=lambda x: x[1],
                         reverse=True))
    for x in range(n_elites):
        elites[x] = zipped[x][0]
    return elites


def insert_elites(population, elites):
    """Insert the elites into the population.

    This function takes the elites of the previous generation and inserts at
    the head of the current generation, replacing what is currently there.
    Args:
        population (list) A list of the genes for every indivudal in a
        population.
        elites (list) A list of the individuals to be inserted into the
                       population.
    Returns:
        A list of the genes for every indivudal in the population with the
        elites added.

    """
    for i in range(len(elites)):
        population[i] = elites[i]
    return population


def solve_sudoku(file_name, pop_size, max_generations=1000, crossover_rate=1,
                 mutation_rate=0.5, tournament_size=2, multi_mutate=False,
                 dual_selector=False, elitism=False):
    """Solve a Sudoku problem using genetic algorithms.

    This function runs the genetic algorithm, using the operators specified,
    until the termination criteria is reached. Then this function will display
    the best Sudoku grid that has been found by the algorithm. Parameters can
    be set by the user and each parameter will have an effect on the efficiency
    and final result. The default arguments have been set to those used in the
    experiments in the report accompanying this program.
    Args:
        file_name (str)         Name of the sudoku file in csv_sudoku.
        pop_size (int)          Number of individuals in the population.
        max_generations (int)   The number of generations to be run before the
                                    algorithm terminates (default 1000).
        crossover_rate (float)  Chance that a crossover will occur (default 1).
        mutation_rate (float)   The chance that a mutation will occur
                                    (default 0.5).
        tournament_size (int)   The number of individuals within each
                                    tournament (default 2).
        multi_mutate (bool)     Set to use the multi mutate operator (default
                                    False).
        dual_selector (bool)    Set to use the dual selection operator (default
                                    False).
        elitism (bool)          Set to use the elitism operator (default
                                    False).
    Returns:
        None

    """
    # Set internal settings
    tournament_select = pop_size
    n_children = pop_size
    duel_selection = 0.8
    n_elites = int(pop_size*0.05)

    # Set generation counter and row limit
    count = 0
    platau_count = 0
    platau_limit = 100

    # Read grid from path
    file_path = join(CSV_PATH, basename(file_name))
    sudoku_grid = read_sudoku(file_path)

    # Initalise the population
    fixed_indicies = np.asarray([0 if n == 0 else 1 for n in sudoku_grid],
                                dtype=np.uint8)
    population = initalise_population(sudoku_grid, pop_size)
    fitnesses = evaluate_population(population)

    best_score = max(fitnesses)
    print("Inital best fitness = ", best_score)

    while (count < max_generations and best_score != 1
           and platau_count != platau_limit):
        prev_best = best_score
        if elitism:
            # Generate a list of elites
            elites = get_elites(population, fitnesses, n_elites)

        # Choose selector
        if dual_selector:
            selection = duel_tournament_selection(population, fitnesses,
                                                  tournament_select,
                                                  tournament_size,
                                                  duel_selection)
        else:
            selection = tournament_selection(population, fitnesses,
                                             tournament_select,
                                             tournament_size)

        # Generate children
        children = create_children(selection, n_children,
                                   crossover_rate=crossover_rate)

        # Select mutator
        if multi_mutate:
            mutants = multi_mutate_offspring(children, mutation_rate,
                                             fixed_indicies)
        else:
            mutants = swap_mutate_offspring(children, mutation_rate,
                                            fixed_indicies)

        if elitism:
            # Add elites to the offsprint
            population = insert_elites(mutants, elites)
        else:
            # Replace population completly
            population = mutants

        fitnesses = evaluate_population(population)
        best_score = max(fitnesses)

        count += 1

        # Check for plateu
        if prev_best == best_score:
            platau_count += 1
        else:
            platau_count = 0
        # Print current best score
        print("Generation", str(count).zfill(4), best_score)

    # On end:
    #  Ensure found grid is valid;
    #  Print status and final Sudoku grid.
    assert check_positions(population[np.argmax(fitnesses)], sudoku_grid)
    if best_score >= 1:
        assert check_valid(population[np.argmax(fitnesses)], sudoku_grid)
        print("Found a solution:")
    elif platau_count == platau_limit:
        print("Local Minima Found. Best solution:")
    else:
        print("No solution found. Best solution:")
    print_sudoku(population[np.argmax(fitnesses)])


if __name__ == "__main__":
    desc = "Solve a Sudoku puzzle using Genetic Algorithms."
    epilog = "For more information please read the README."

    # Create argument parser
    parser = argparse.ArgumentParser(description=desc, epilog=epilog)
    parser.add_argument('-i', '--input', help="name of file to run",
                        required=True)
    parser.add_argument('-p', '--population', help="inital population size",
                        required=True, type=int)
    parser.add_argument('-m', '--multi_mutate', help="use multi mutate",
                        required=False, action='store_true')
    parser.add_argument('-e', '--elitism', help="use elitism",
                        required=False, action='store_true')
    parser.add_argument('-d', '--duel_selector', help="use duel selector",
                        required=False, action='store_true')
    args = parser.parse_args()

    solve_sudoku(args.input, int(args.population), elitism=args.elitism,
                 multi_mutate=args.multi_mutate,
                 dual_selector=args.duel_selector)
