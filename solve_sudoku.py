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
"""

from sudoku_utils import read_sudoku

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


def get_boxes(sudoku_gene):
    """Return a list representing the 3x3 boxes.

    This function converts a 1 dimensional list which represents the sudoku
    gene and converts it into a list of the 3x3 boxes represented as lists.

    args:
        sudoku_gene (list)  List representing the sudoku grid.
    return:
        A two dimensional list of sudoku boxes.
    """
    box_array = [[[]]]
    for i, number in enumerate(sudoku_gene):
        box_array[int(i/9/3), (i % 9 % 3)].append(number)
    return sum(box_array, [])


def get_rows(sudoku_gene):
    """Return a list representing the rows.

    This function converts a 1 dimensional list which represents the sudoku
    gene and converts it into a list of the columns represented as lists.

    args:
        sudoku_gene (list)  List representing the sudoku grid.
    return:
        A two dimensional list of sudoku columns.
    """
    row_array = [[]]
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
    column_array = [[]]
    for i, number in enumerate(sudoku_gene):
        column_array[i % 9].append(number)
    return column_array


if __name__ == "__main__":
    gene = read_sudoku("./csv_sudoku/Grid1.csv")
    print(gene)
