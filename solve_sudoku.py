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
from random import randint, random, sample
import numpy as np

CSV_PATH = "./csv_sudoku"
OUT_PATH = "./solved_sudoku"


# Datatype is always np.uint8
# Worse fitness is 243
# Elitist Generational Genetic Algorithm
# Fitness Function
#  Minimise the number of non-unique numbers in each row, column, and 3x3 box
#  0 being the best and infinity the worst
# Selection Criteria
#  Tournament selection to generate a 2*(popsize-1) collection of parents
#  set t_size to 1/5 of population size
#  Could be replaced with rank based selection if the tournmanet size is an
#  issue with the low population sizes?
# Crossover
#  Binary Crossover - since crossing over using this method ignores the
#  imovable bits. Set the crossover change to 0.3?
# Mutation Operator:
#  M-gene mutation based on chance? Or just one gene?
#  Swap mutation is bad because it may not remove duplicate numbers?
# Replacement
#  Keep the best in the og population and replace all the rest with children?
# Termination Criteria:
#  Early stopping, ten(?) generations of no change
#  If the best solution has a fitness function of 0
#
# Keep a list of the indexes which were og 0 and are not allowed to be changed.


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


def initalise_population(sudoku_gene, fixed_indicie, pop_size):
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
    boxes = get_boxes(sudoku_gene)
    rows = get_rows(sudoku_gene)
    columns = get_columns(sudoku_gene)

    for box in boxes:
        score += len(set(box))
    for row in rows:
        score += len(set(row))
    for column in columns:
        score += len(set(column))

    return score/243


def evaluate_population(population_array):
    """Evaluate the fitness score of each individual in a population."""
    fitness_array = np.zeros((population_array.shape[0]))
    for i, individual in enumerate(population_array):
        fitness_array[i] = evaluate_individual(individual)
    return fitness_array


def tournament_selection(population_array, fitness_array, t_size, t_select):
    """Select parents from the population using tournament selection."""
    parents = np.zeros((t_select), dtype=np.uint32)
    for i in range(t_select):
        tournament_list = []
        for _ in range(t_size):
            tournament_list.append(randint(0, population_array.shape[0]-1))
        tournament_fitnesses = [fitness_array[i] for i in tournament_list]
        parents[i] = tournament_list[np.argmax(tournament_fitnesses)]
    return parents


def random_selection(population_array, fitness_array, t_size, t_select):
    """Select parents from the population using random selection."""
    parents = np.zeros((t_select), dtype=np.uint32)
    for i in range(t_select):
        parents[i] = randint(0, len(population_array)-1)
    return parents


def crossover(parent_1, parent_2, crossover_rate):
    """Create a single offspring gene from two parent genees.

    Talk about binary mask crossover.
    """
    offspring1 = []
    offspring2 = []
    binary_mask = [1 if random() < crossover_rate
                   else 0 for _ in range(len(parent_1))]
    for i in range(len(parent_1)):
        if binary_mask[i]:
            offspring1.append(parent_1[i])
            offspring2.append(parent_2[i])
        else:
            offspring1.append(parent_2[i])
            offspring2.append(parent_1[i])
    return offspring1, offspring2


def create_offspring(population_array, parents, crossover_rate):
    """Create offspring using by crossover from pairs of parents."""
    offspring = []
    for i in range(int(len(parents)/2)):
        parent_1 = -1
        parent_2 = -1
        while parent_1 == parent_2:
            parent_1 = randint(0, parents.shape[0]-1)
            parent_2 = randint(0, parents.shape[0]-1)
        parent_1_gene = population_array[parents[parent_1]]
        parent_2_gene = population_array[parents[parent_2]]
        child_1, child_2 = crossover(parent_1_gene, parent_2_gene,
                                     crossover_rate)
        offspring.append(child_1)
        offspring.append(child_2)
    return np.asarray(offspring, dtype=np.uint8)


def old_mutate_offspring(offspring_array, mutation_rate, fixed_indicies):
    """Mutate all the offspring to produce distrinct offspring."""
    for i, child in enumerate(offspring_array):
        mutant_child = []
        for j in range(len(child)):
            if fixed_indicies[j]:
                mutant_child.append(child[j])
            else:
                if random() < mutation_rate:
                    mutant_child.append(randint(1, 9))
                else:
                    mutant_child.append(child[j])
        offspring_array[i] = mutant_child
    return offspring_array


def swap_mutate_offspring(offspring_array, mutation_rate, fixed_indicies):
    """Mutate all the offspring to produce distrinct offspring."""
    for i, child in enumerate(offspring_array):
        if random() < mutation_rate:
            swap_index_1 = randint(0, len(child)-1)
            swap_index_2 = randint(0, len(child)-1)
            while fixed_indicies[swap_index_1] or fixed_indicies[swap_index_2]:
                swap_index_1 = randint(0, len(child)-1)
                swap_index_2 = randint(0, len(child)-1)
            swap_temp = child[swap_index_1]
            child[swap_index_1] = child[swap_index_2]
            child[swap_index_2] = swap_temp
        offspring_array[i] = child
    return offspring_array


def m_gene_mutate_offspring(offspring_array, mutation_rate, fixed_indicies):
    """Mutate all the offspring to produce distrinct offspring."""
    m = 1
    for i, child in enumerate(offspring_array):
        if random() < mutation_rate:
            for _ in range(m):
                mutate_index = randint(0, len(child)-1)
                while fixed_indicies[mutate_index]:
                    mutate_index = randint(0, len(child)-1)
                child[mutate_index] = randint(1, 9)
        offspring_array[i] = child
    return offspring_array


def solved_sudoku(file_name, pop_size):
    """Solve a Sudoku problem using genetic algorithms.

    args:
        file_name (str) Name of the sudoku file in csv_sudoku.
        pop_size (int)  Number of individuals in the population.
    """
    max_generations = 1000
    crossover_rate = 0.1
    mutation_rate = 0.5
    tournmanet_size = int(pop_size/20)+1
    n_parents = pop_size

    file_path = join(CSV_PATH, basename(file_name))
    sudoku_grid = read_sudoku(file_path)

    fixed_indicies = np.asarray([1 if n == 0 else 0 for n in sudoku_grid],
                                dtype=np.uint8)
    population_array = initalise_population(sudoku_grid, fixed_indicies,
                                            pop_size)

    print("Population Initalised")

    fitness_array = evaluate_population(population_array)
    best_index = np.argmax(fitness_array)
    best_fitness = fitness_array[best_index]
    best_gene = population_array[best_index]
    print("Best inital fitness: ", best_fitness)

    generation = 0
    while generation < max_generations and best_fitness < 1:
        parents = tournament_selection(population_array, fitness_array,
                                       tournmanet_size, n_parents)
        # parents = random_selection(population_array, fitness_array,
        #                            tournmanet_size, n_parents)
        children = create_offspring(population_array, parents, crossover_rate)
        # print(len(children))
        population_array = swap_mutate_offspring(children, mutation_rate,
                                                 fixed_indicies)
        print(len(population_array))
        fitness_array = evaluate_population(population_array)
        # worst_index = np.argmin(fitness_array)
        # fitness_array[worst_index] = best_fitness
        # population_array[worst_index] = best_gene

        best_index = np.argmax(fitness_array)
        best_fitness = fitness_array[best_index]
        best_gene = population_array[best_index]
        print("Generation:", generation, "=", best_fitness)
        generation += 1
    print_sudoku(best_gene)


if __name__ == "__main__":
    # gene = read_sudoku("./csv_sudoku/Grid1.csv")
    # print(gene)
    # for box in get_boxes(gene):
    #     print(box)
    # for row in get_rows(gene):
    #     print(row)
    solved_sudoku("Grid3.csv", 1000)
