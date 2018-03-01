"""Automate a number of tests for use in the report.

This program uses a Genetic Algorithm to solve a number of given preset
Sudoku problems at a range of innital population sizes while printing the
results as it does so.

Flags:
    -i, --input         Name of the grid file in csv_sudoku to solve.
    -p, --population    Population size.
Example:
    $ python convert_grid.py -i Grid1.csv -p 10

"""

from solve_sudoku_clean import solve_sudoku
import sys
import os
import argparse


if __name__ == "__main__":
    desc = "Solve a Sudoku puzzle using Genetic Algorithm for a range of pops."
    n_repeats = 5
    populations = [10, 100, 1000, 10000]

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input', help="Name of file.",
                        required=True)
    args = parser.parse_args()
    sudoku = args.input

    for population in populations:
        out_name = os.path.splitext(sudoku)[0]+"_"+str(population)+".txt"
        print(out_name)
        out_path = os.path.join("./logs", out_name)
        orig_stdout = sys.stdout
        with open(out_path, 'w+') as f:
            for i in range(n_repeats):

                    sys.stdout = f
                    print()
                    print("File:", sudoku, " Pop:", population, " trail:", i+1)
                    solve_sudoku(sudoku, population)
                    print("************")
                    print()
        sys.stdout = orig_stdout
