"""Library of Sudoku utilities.

Contains functions shared across the Genetic Algorithm Sudoku project.

Example:
    from sudoku_utils import print_sudoku
"""
import csv


def read_sudoku(input_path):
    """Read a sudoku grid from a csv file.

    This function reads a sudoku grid as a one dimensional sudoku grid from
    the specified csv file. The csv file must not contain any spaces or extra
    white space at the end of each line. Additionally each number must be in
    a seperate cell with 0 representing blanks.
    args:
        output_path (str)    Path of the file to be written to.
    return:
        One dimensional list representation of a sudoku gene.
    """
    sudoku_gene = []
    with open(input_path, "r", newline='') as input_file:
        reader = csv.reader(input_file)
        for row in reader:
            for number in row:
                sudoku_gene.append(int(number))
    return sudoku_gene


def save_sudoku(sudoku_gene, output_path):
    """Save a Sudoku gene list as a csv.

    This procedure saves the genetic representation of the sudoku grid and
    saves it as a 9x9 grid within a csv file. Due to the limitations of the
    file format and for ease of use the function omits the boundry lines for
    the 3x3 subgrids as well as the outer border lines of the grid.

    args:
        sudoku_gene (list)   Genetic representation of a Sudoku grid.
        output_path (str)    Path of the file to be written to.
    """
    with open(output_path, "w+", newline='') as output_file:
        csv_writer = csv.writer(output_file, delimiter=",")
        number_line = []
        for i, number in enumerate(sudoku_gene):
            number_line.append(number)
            if (i + 1) % 9 == 0:
                csv_writer.writerow(number_line)
                number_line = []


def print_row(row):
    """Print each char in a list.

    This procedure prints each charecter in a given list and then appends an
    end line to the end while flushing the input buffer.

    args:
        row (list)         List representing a row of a Sudoku grid
    """
    for c in row:
        print(c, end=" ", flush=True)
    print()


def print_sudoku(sudoku_gene):
    """Print Sudoku genes as Sudoku grids.

    This procedure takes a Sudoku grid encoded as a list of 81 numbers, with
    0 representing blanks and prints it to console in the traditional 9x9
    grid format with accompanying 3x3 boxes.

    args:
        sudoku_gene (list)  Genetic representation of a Sudoku grid.
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
