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
    $ python convert_grid.py -i Grid1.csv -p 10

ToDo:
    Explain the grid indexing to produce rows, columns, boxes.
    Finish docstrings, add arguments and returns
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

    args:
        sudoku_gene (list)  List representing the sudoku grid.
    return:
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

    args:
        sudoku_gene (list)  List representing the sudoku grid.
    return:
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

    args:
        sudoku_gene (list)  List representing the sudoku grid.
    return:
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
    """Check that the preset grid numbers are still present in the original."""
    correct = True
    for i in range(len(original)):
        if original[i] != 0 and original[i] != to_check[i]:
            print("Incorrect index:", i)
            correct = False
    return correct


def check_valid(to_check, original):
    """Check that the preset grid numbers form a valid solution."""
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

    args:
        sudoku_gene (list)      List representing the sudoku grid.
        fixed_indicies (List)   Indexes which are not to be changed.
        pop_size (int)          Number of individuals in the population.

    return:
        A list of randomly generated Sudoku solutions.
    """
    population_array = np.zeros((pop_size, len(sudoku_gene)), dtype=np.uint8)
    for i in range(pop_size):
        new_pop = get_rows(sudoku_gene)
        for j, box in enumerate(new_pop):
            rand_set = [i for i in range(len(box)+1)]
            for n in set(box):
                rand_set.remove(n)
            for k, num in enumerate(box):
                if num == 0:
                    rand_int = sample(rand_set, 1)[0]
                    new_pop[j][k] = rand_int
                    rand_set.remove(rand_int)

        population_array[i] = sum(new_pop, [])
        # for j in range(len(sudoku_gene)):
        #     if sudoku_gene[j] == 0:
        #         population_array[i][j] = randint(1, 9)
        #     else:
        #         population_array[i][j] = sudoku_gene[j]
    return population_array


def evaluate_individual(sudoku_gene):
    """Get the fitness score of an individual within a population."""
    score = 0
    # boxes = get_boxes(sudoku_gene)
    # rows = get_rows(sudoku_gene)
    # columns = get_columns(sudoku_gene)

    for box in get_boxes(sudoku_gene):
        score += len(set(box))
    # for row in rows:
    #     score += len(set(row))
    for column in get_columns(sudoku_gene):
        score += len(set(column))

    return score/162


def evaluate_population(population_array):
    """Evaluate the fitness score of each individual in a population."""
    fitness_array = np.zeros((population_array.shape[0]))
    for i, individual in enumerate(population_array):
        fitness_array[i] = evaluate_individual(individual)
    return fitness_array


def tournament_selection(population_array, fitness_array, t_select, t_size):
    """Select parents from the population using tournament selection."""
    parents = np.zeros((t_select, population_array.shape[1]), dtype=np.uint32)
    for i in range(t_select):
        # pop_zip = list(zip(population_array, fitness_array))
        # tournament_list = sample(pop_zip, t_size)
        # shuffle(tournament_list)
        # genes, fitnesses = zip(*tournament_list)
        # winner_index = np.argmax(fitnesses)
        # parents[i] = genes[winner_index]
        tournament_list = sample([j for j in range(len(population_array))],
                                 t_size)
        shuffle(tournament_list)
        t_fitnesses = [fitness_array[j] for j in tournament_list]
        winner_index = tournament_list[np.argmax(t_fitnesses)]
        parents[i] = population_array[winner_index]
    return parents


def row_crossover(parent_1, parent_2, crossover_rate):
    """Create two offspring genes from two parent genees.

    Talk about binary mask crossover.
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


def create_children(parents, crossover_rate, n_children):
    """Create offspring using by crossover from pairs of parents."""
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
    """Mutate all the offspring to produce distrinct offspring."""
    for i, child in enumerate(offspring_array):
        if random() < mutation_rate:
            mutant = child.copy()
            rand_row = randrange(0, len(mutant), 9)
            swap_index_1 = rand_row + randint(0, 8)
            swap_index_2 = rand_row + randint(0, 8)
            while (swap_index_1 == swap_index_2 or fixed_indicies[swap_index_1]
                   or fixed_indicies[swap_index_2]):
                swap_index_1 = rand_row + randint(0, 8)
                swap_index_2 = rand_row + randint(0, 8)
            swap_temp = mutant[swap_index_1]
            mutant[swap_index_1] = mutant[swap_index_2]
            mutant[swap_index_2] = swap_temp
            offspring_array[i] = mutant
    return offspring_array


def solve_sudoku(file_name, pop_size, max_generations=10000, crossover_rate=1,
                 mutation_rate=0.5, tournament_size=2):
    """Solve a Sudoku problem using genetic algorithms.

    args:
        file_name (str) Name of the sudoku file in csv_sudoku.
        pop_size (int)  Number of individuals in the population.
    """
    tournament_select = pop_size
    n_children = pop_size
    count = 0
    platau_count = 0
    platau_limit = 1000

    file_path = join(CSV_PATH, basename(file_name))
    sudoku_grid = read_sudoku(file_path)
    fixed_indicies = np.asarray([0 if n == 0 else 1 for n in sudoku_grid],
                                dtype=np.uint8)
    population = initalise_population(sudoku_grid, pop_size)
    fitnesses = evaluate_population(population)
    best_score = max(fitnesses)
    print("Inital best fitness = ", best_score)

    while (count < max_generations and best_score != 1
           and platau_count != platau_limit):
        prev_best = best_score
        selection = tournament_selection(population, fitnesses,
                                         tournament_select, tournament_size)
        children = create_children(selection, crossover_rate, n_children)
        population = swap_mutate_offspring(children, mutation_rate,
                                           fixed_indicies)
        # population = mutants
        fitnesses = evaluate_population(population)
        best_score = max(fitnesses)
        count += 1
        if prev_best == best_score:
            platau_count += 1
        else:
            platau_count = 0
        print("Generation", str(count).zfill(4), best_score)

    if best_score >= 1:
        print("Found a solution:")
        assert check_valid(population[np.argmax(fitnesses)], sudoku_grid)
    elif platau_count == platau_limit:
        print("Local Minima Found. Best solution:")
    else:
        print("No solution found. Best solution:")
    print_sudoku(population[np.argmax(fitnesses)])


if __name__ == "__main__":
    desc = "Solve a Sudoku puzzle using Genetic Algorithms."
    epilog = "For more information please read the README."

    # Create Argument parset
    parser = argparse.ArgumentParser(description=desc, epilog=epilog)
    parser.add_argument('-i', '--input', help="Name of file.",
                        required=True)
    parser.add_argument('-p', '--population', help="Population size.",
                        required=True, type=int)
    args = parser.parse_args()
    solve_sudoku(args.input, int(args.population))
