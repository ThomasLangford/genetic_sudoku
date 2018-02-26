"""Library of Sudoku utilities.

Contains functions shared across the Genetic Algorithm Sudoku project.

Example:
    from sudoku_utils import print_sudoku
"""


def print_row(row):
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
                hash_line = ["-" for i in range(11)]
                print_row(hash_line)
            row = []
        elif (i + 1) % 3 == 0:
            row.append("|")
