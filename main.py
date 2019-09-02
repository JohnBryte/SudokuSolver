# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 13:48:35 2019

@author: Johannes
"""

from SudokuExtractor import *
from Solver import *
import os

if __name__ == '__main__':
    puzzle = SudokuExtractor()
    print("WELCOME TO MY SUDOKU SOLVER")
    while(True):
        #print("Do you want to solve a Sudoku? [y]es [n]o")
        solve = input("Do you want to solve a Sudoku? [y]es [n]o \n")
        if solve == 'n':
            break
        else:
            img_dir = input("What is the filename? E.g.: 'image12.jpg'\n")
            puzzle.load_image(os.path.join('pics', img_dir))
            puzzle.extract_puzzle()
#            puzzle.show_extraction()
            try:
                solver = Solver(puzzle.grid)
                proceed = input("Solve puzzle? [y]es [n]o?\n")
                if proceed == 'y':
                    solver.solve()
                    solver.print_sudoku()
            except:
                print("Please try another puzzle.")
    print("Bye")