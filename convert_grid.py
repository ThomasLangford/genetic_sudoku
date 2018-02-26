"""Convert the raw .ss grid files to .csv.

This program converts a single Sudoku grid in the raw_sudoku folder and then
converts it to a more user friendly and pythonic csv file for use later. The
output file has the same name as the input file with a csv extension and will
be saved in the csv_sudoku folder

Flags:
    -i --input Name of the file in raw_sudoku to convert.
Example:
    $ python convert_grid.py -i Grid1.ss

"""
