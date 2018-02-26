"""Library of Sudoku utilities.

Contains functions shared across the Genetic Algorithm Sudoku project.

Example:
    from sudoku_utils import print_sudoku
"""
import csv


def print_row(row):
    """Print each char in a list.

    args:
        row (list) list representing a row of a Sudoku grid
    """
    for c in row:
        print(c, end=" ", flush=True)
    print()


def print_sudoku(sudoku_gene):
    """Print Sudoku genes as grids.

    args:
        sudoku_gene (list) Genetic representation of a Sudoku grid.
    """
    row = []
    row_count = 0
    for i, number in enumerate(sudoku_gene):
        row.append(number)
        if (i + 1) % 9 == 0:
            print_row(row)
            row_count += 1
            if (row_count) % 3 == 0 and row_count < 9:
                dash_line = ["-" for i in range(11)]
                print_row(dash_line)
            row = []
        elif (i + 1) % 3 == 0:
            row.append("|")


def save_sudoku(sudoku_gene, output_path):
    """Save Sudoku as a csv."""
    with open(output_path, "w+", newline='') as output_file:
        csv_writer = csv.writer(output_file, delimiter=",")
        number_line = []
        for i, number in enumerate(sudoku_gene):
            number_line.append(number)
            if (i + 1) % 9 == 0:
                csv_writer.writerow(number_line)
                number_line = []
