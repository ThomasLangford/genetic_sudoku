"""Solve a given Sudoku problem using Genetic Algorithms.

This program uses a Genetic Algorithm to solve a specified Sudoku problem
contained within a csv file. The found solution or best solution if the
stopping criteria has been reached will be displayed on screen along with the
fitness score. It will then be saved as a csv file in the solved_sudoku folder.

Flags:
    -i, --input         Name of the grid file in csv_sudoku to solve.
    -p, --population    Population size.
Example:
    $ python convert_grid.py -i Grid1.csv -p 10
"""
# Elitist Generational
# Selection Criteria
#   Tournament selection to generate a 2*(popsize-1) collection of parents
# Crossover
#  Binary Crossover - since crossing over using this method ignores the
#  imovable bits. Set the crossover change to 0.3?
# Mutation Operator:
#  M-gene mutation based on chance? Or just one gene?
#  Swap mutation is bad because it may not remove duplicate numbers?
# Termination Criteria:
#  Early stopping, ten(?) generations of no change
#  If the best solution has a fitness function of 0
