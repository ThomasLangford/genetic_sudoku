"""Convert the raw .ss or .txt grid files to .csv.

This program converts a single Sudoku grid in the raw_sudoku folder and then
converts it to a more user friendly and pythonic csv file for use later. The
output file has the same name as the input file with a csv extension and will
be saved in the csv_sudoku folder

Flags:
    -i, --input         Name of the file in raw_sudoku to convert.
Example:
    $ python convert_grid.py -i Grid1.ss

ToDo:
    Import sudoku print method to print out the final array before saving.
"""

import argparse
import csv
from os.path import basename, join, splitext


def convert_sudoku_to_csv(file_name):
    """Convert a .txt or .ss file to csv.

    This funtion converts a file to csv format and saves the resulting output.
    Blank squares in the Sudoku grid are represented as either the charecters
    '0' or '.'. Any other non integer charecters are ignored. The file will
    be saved in the ./csv_sudoku folder.

    Args:
        file_name (str) Name of the file in ./raw_sudoku to be converted.

    """
    csv_name = splitext(basename(file_name))[0] + ".csv"

    # Set input and output file paths
    input_path = join("./raw_sudoku", basename(file_name))
    output_path = join("./csv_sudoku", csv_name)

    # Open input file and create output file
    with open(input_path, "r") as input_file:
        with open(output_path, "w+") as output_file:
            csv_writer = csv.writer(output_file, delimiter=",")
            for i, line in enumerate(input_file):
                # Drop lines begining with "-"
                if line[0] == "-":
                    continue
                number_line = []
                # Convert numerics to int or add 0 to represent blanks
                for char in line:
                    if char.isdigit():
                        number_line.append(int(char))
                    elif char == ".":
                        number_line.append(0)
                # Ensure that the line length = 9
                assert len(number_line) == 9
                csv_writer.writerow(number_line)


if __name__ == "__main__":
    # Set argparse text
    desc = "Convert .ss or .txt Sudoku files to .csv."
    epilog = "For more information please read the README."

    # Create Argument parset
    parser = argparse.ArgumentParser(description=desc, epilog=epilog)
    parser.add_argument('-i', '--input', help="Name of file.", required=True)
    args = parser.parse_args()
    convert_sudoku_to_csv(args.input)
