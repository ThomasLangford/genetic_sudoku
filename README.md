# genetic_sudoku
Using genetic algorithms to solve a game of Sudoku.
The Sudoku puzzle is a combinatorial number placement problem where the player must fill a partially completed nine by nice grid with sets of the numbers one to nine. These numbers must be placed in a way such that each row, column, and three by three sub grid contains a unique set. Due to this constraint on the placement of numbers, Sudoku proves to be a NP-complete problem which makes it impracticable to solve using a direct method. Instead a heuristic search method must be applied to the problem, of which the genetic algorithm metaheuristic is implemented here.

Examples of the Sudoku format as both .csv and .ss files have been placed in the appropriate folders. Additionally, the .csv grid files used to produce the results contained within the report have been saved in the csv_sudoku folder.

## Contents:
1. Working Folder Layout
2. Converting from .SS to .CSV
3. Sudoku .CSV Format
4. Running the Genetic Algorithm
5. Replicating the Experiments


## 1.   Working Folder Layout
The format of the working directory should be as follows:

    genetic_sudoku/
        csv_sudoku/
            Grid1.csv
            ...
        raw_sudoku/
            Grid1.ss
            ...
        ...
        
## 2.   Converting from .SS to .CSV 
To convert from a Sudoku grid in the .ss format to the .csv format accepted by this program, the convert_grid program should be used. First place the .ss file in the ./raw_sudoku folder as <name>.ss. Then run the following command:
        
        python convert_grid.py -i <name>.ss
    
## 3.   Sudoku .CSV Format
If you wish to write your own sudoku grid you must adhere to the following formatting:
1. All values in the Sudoku grid must be seperated by commas.
2. All clues must be represented by their numerical equivalent. 
3. All blank spaces must be represnted by 0.
4. No trailing or leading empty cells.
5. The grid must be saved as a .csv file.

An example has been provided below.

    4,0,0,1,0,2,6,3,0
    5,0,0,6,4,3,8,0,0
    7,6,0,5,0,8,4,1,2
    6,0,0,0,0,9,3,4,8
    2,4,0,8,3,0,9,5,0
    8,0,9,4,1,5,0,7,0
    0,7,2,0,0,4,0,6,0
    0,5,4,2,0,0,0,8,9
    0,8,6,3,0,7,1,2,4
    
## 4.    Running the Genetic Algorithm
To run the genetic algorithm you must select the inital population size as well as the input grid. Additionally you can set any number of flags to add advanced operators to the genetic algorithm. For example, to run the the grid contained in <name>.csv with a population of n you must run the following command:

    python solve_sudoku.py -i <name>.csv -p n

### Flags:

    usage: solve_sudoku.py [-h] -i INPUT -p POPULATION [-m] [-e] [-d]
    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT, --input INPUT
                            name of file to run
      -p POPULATION, --population POPULATION
                            inital population size
      -m, --multi_mutate    use multi mutate
      -e, --elitism         use elitism
      -d, --duel_selector   use duel selector
      
## 5.   Replicating the Experiments
Each set of commands was run five times and the average taken.
### Experiment 1
#### Grid 1

    python solve_sudoku.py -i Grid1.csv -p 10 &
    python solve_sudoku.py -i Grid1.csv -p 100 &
    python solve_sudoku.py -i Grid1.csv -p 1000 &
    python solve_sudoku.py -i Grid1.csv -p 10000
    
#### Grid 2

    python solve_sudoku.py -i Grid2.csv -p 10 &
    python solve_sudoku.py -i Grid2.csv -p 100 &
    python solve_sudoku.py -i Grid2.csv -p 1000 &
    python solve_sudoku.py -i Grid2.csv -p 10000

#### Grid 3

    python solve_sudoku.py -i Grid3.csv -p 10 &
    python solve_sudoku.py -i Grid3.csv -p 100 &
    python solve_sudoku.py -i Grid3.csv -p 1000 &
    python solve_sudoku.py -i Grid3.csv -p 10000
    
### Experiment 2
#### Population = 100
    
    python solve_sudoku.py -i Grid3.csv -p 100 -d
            
