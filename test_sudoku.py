"""Automate a number of tests for use in the report.

This program uses a Genetic Algorithm to solve a given Sudoku problems at a
range of innital population sizes while saving the results to a log file as it
does so.

Flags:
    -i, --input         Name of the grid file in csv_sudoku to solve.
Example:
    $ test_sudoku.py -i Grid3.csv

"""

from solve_sudoku_clean import solve_sudoku
import sys
import os
import argparse


if __name__ == "__main__":
    desc = "Solve a Sudoku puzzle using Genetic Algorithm for a range of pops."
    n_repeats = 5
    populations = [10, 100, 1000, 10000]
    m_mutate = True
    elitism = False
    dual_selector = False

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input', help="Name of file.",
                        required=True)
    args = parser.parse_args()
    sudoku = args.input

    for population in populations:
        out_name = os.path.splitext(sudoku)[0]+"_"+str(population)+".txt"
        print(out_name)
        out_path = os.path.join("./logs", out_name)

        with open(out_path, 'w+') as f:
            f.write("Mutli_mutate:", m_mutate)
            f.write("Elitism:", elitism)
            f.write("Dual_selector:", dual_selector)
            for i in range(n_repeats):
                    print(i)
                    orig_stdout = sys.stdout
                    sys.stdout = f
                    print()
                    print("File:", sudoku, " Pop:", population, " trail:", i+1)
                    solve_sudoku(sudoku, population, max_generations=1000,
                                 multi_mutate=m_mutate, elitism=elitism,
                                 dual_selector=dual_selector)
                    print("************")
                    print()
                    sys.stdout = orig_stdout
    # test_sudoku.py -i Grid3.csv & shutdown -s
