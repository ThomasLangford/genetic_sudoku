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
import time


if __name__ == "__main__":
    desc = "Solve a Sudoku puzzle using Genetic Algorithm for a range of pops."
    n_repeats = 5
    populations = [10, 100, 1000, 10000]
    max_g = 1000
    # populations = [10000]

    local_time = time.localtime()
    time_string = time.strftime("%Y_%m_%d_%H_%M", local_time)

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input', help="Name of file.",
                        required=True)
    parser.add_argument('-m', '--multi_mutate', help="Use multi mutate.",
                        required=False, action='store_true')
    parser.add_argument('-e', '--elitism', help="Use elitism",
                        required=False, action='store_true')
    parser.add_argument('-d', '--duel_selector', help="Name of file.",
                        required=False, action='store_true')
    args = parser.parse_args()

    sudoku = args.input
    m_mutate = args.multi_mutate
    elitism = args.elitism
    dual_selector = args.duel_selector

    out_dir = os.path.join("./logs", time_string)
    print(out_dir)
    print("Mutli_mutate: "+str(m_mutate)+" ")
    print("Elitism: "+str(elitism)+" ")
    print("Dual_selector: "+str(dual_selector)+" ")
    print()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for population in populations:
        out_name = os.path.splitext(sudoku)[0]+"_"+str(population)+".txt"
        print(out_name)
        out_path = os.path.join(out_dir, out_name)
        if population == 10000:
            max_g_c = 150
        else:
            max_g_c = max_g
        with open(out_path, 'w+') as f:
            f.write("Mutli_mutate: "+str(m_mutate)+" ")
            f.write("Elitism: "+str(elitism)+" ")
            f.write("Dual_selector: "+str(dual_selector)+" ")
            for i in range(n_repeats):
                    print(i)
                    orig_stdout = sys.stdout
                    sys.stdout = f
                    print()
                    print("File:", sudoku, " Pop:", population, " trail:", i+1)
                    solve_sudoku(sudoku, population, max_generations=max_g_c,
                                 multi_mutate=m_mutate, elitism=elitism,
                                 dual_selector=dual_selector)
                    print("************")

                    print()
                    sys.stdout = orig_stdout
