"""Solve a given Sudoku problem using Genetic Algorithms.

This program uses a Genetic Algorithm to solve a specified Sudoku problem
contained within a csv file. The found solution or best solution if the stopping
criteria has been reached will be displayed on screen along with the fitness
score. It will then be saved as a csv file in the solved_sudoku folder.

Flags:
    -i --input Name of the grid file in csv_sudoku to solve.
    -p --population Population size.
Example:
    $ python convert_grid.py -i Grid1.csv -p 10
"""
