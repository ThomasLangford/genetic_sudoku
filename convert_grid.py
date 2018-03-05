"""Convert the raw .ss or .txt grid files to .csv.

This program converts a single Sudoku grid in the raw_sudoku folder and then
converts it to a more user friendly and pythonic csv file for use later. The
output file has the same name as the input file with a csv extension and will
be saved in the csv_sudoku folder

Flags:
    -i, --input         Name of the file in raw_sudoku to convert.
Example:
    $ python convert_grid.py -i Grid1.ss

"""

import argparse
from os.path import basename, join, splitext
from sudoku_utils import print_sudoku, save_sudoku

SS_PATH = "./raw_sudoku"
CSV_PATH = "./csv_sudoku"


def convert_sudoku_to_csv(file_name, view=False):
    """Convert a .txt or .ss file to csv.

    This funtion converts a file to csv format and saves the resulting output.
    Blank squares in the Sudoku grid are represented as either the charecters
    '0' or '.'. Any other non integer charecters are ignored. The file will
    be saved in the ./csv_sudoku folder.

    Args:
        file_name (str) Name of the file in ./raw_sudoku to be converted.
        view (bool)     Set to display preview. (default False)
    Returns:
        None

    """
    csv_name = splitext(basename(file_name))[0] + ".csv"
    sudoku_grid = []

    # Set input and output file paths
    input_path = join(SS_PATH, basename(file_name))
    output_path = join(CSV_PATH, csv_name)

    # Open input file and create output file
    with open(input_path, "r") as input_file:
        for i, line in enumerate(input_file):
            # Drop lines begining with "-"
            if line[0] == "-":
                continue
            # Convert numerics to int or add 0 to represent blanks
            for char in line:
                if char.isdigit():
                    sudoku_grid.append(int(char))
                elif char == ".":
                    sudoku_grid.append(0)
    # Ensure that grid contains 81 numbers
    assert len(sudoku_grid) == 81, "Incorrect total of numbers in grid."
    if view:
        print_sudoku(sudoku_grid)
    save_sudoku(sudoku_grid, output_path)


if __name__ == "__main__":
    # Set argparse text
    desc = "Convert .ss or .txt Sudoku files to .csv."
    epilog = "For more information please read the README."

    # Create Argument parset
    parser = argparse.ArgumentParser(description=desc, epilog=epilog)
    parser.add_argument('-i', '--input', help="name of file to convert",
                        required=True)
    args = parser.parse_args()

    # Convert sudoku grid to csv
    convert_sudoku_to_csv(args.input, view=True)
