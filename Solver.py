# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 19:52:04 2019

@author: Johannes
"""

class Solver:
    digits = '123456789'
    solved_grid = None

    def __init__(self, grid):
        self.coords, self.peers, self.units, self.all_units = self.sudoku_elements()
        grid = self.parse_grid(grid)
        if self.is_valid(grid):
            self.grid = grid
            print('This is the grid I detected: ')
            self.print_sudoku(self.convert_grid(self.grid))
            
        else:
            print("I'M SORRY DAVE.\nNON VALID SUDOKU - DIGITS NOT CORRECTLY RECOGNISED")
            #raise Exception("I'M SORRY DAVE.\nNON VALID SUDOKU - DIGITS NOT CORRECTLY RECOGNISED")
            self.print_sudoku(self.convert_grid(self.grid))
        
    def load_new_grid(self, grid):
        grid = self.parse_grid(grid)
        if self.is_valid(grid):
            self.grid = grid
        else:
            print("NON VALID SUDOKU - DIGITS NOT CORRECTLY RECOGNISED")
            self.print_sudoku(self.convert_grid(self.grid))
            
    def cross(self, A, B):
        return [a+b for a in A for b in B]
    
    def sudoku_elements(self):
        all_rows = 'ABCDEFGHI'
        all_cols = '123456789'
        coords = self.cross(all_rows, all_cols)  # Flat list of all possible squares

        # Flat list of all possible units
        all_units = [self.cross(row, all_cols) for row in all_rows] + \
                    [self.cross(all_rows, col) for col in all_cols] + \
                    [self.cross(rid, cid) for rid in ['ABC', 'DEF', 'GHI'] for cid in ['123', '456', '789']]  # Squares

        # Indexed dictionary of units for each square (list of lists)
        # Each position will have three units, each a list of 9: row, column and square
        units = {pos: [unit for unit in all_units if pos in unit] for pos in coords}

        # Indexed dictionary of peers for each square (set)
        # Peers are the unique set of possible positions except itself
        peers = {pos: set(sum(units[pos], [])) - {pos} for pos in coords}

        return coords, peers, units, all_units

    def parse_grid(self, grid):
        temp = [str(val) if str(val) in self.digits else '0' for row in grid for val in row]
        if len(temp) == 81:
            return dict(zip(self.coords, temp))
    
    def is_valid(self, grid):
        #checks if grid from digit-recognition is valid
        for cell, value in grid.items():
            if value != '0':
                for peer in self.peers[cell]:
                    if grid[peer] == value:
                        return False
        return True

    def validate_sudoku(self, grid):
        complete = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        return all([sorted([cell for cell in unit]) == complete for unit in self.all_units])
    
    def solve_puzzle(self, puzzle):

        coords, peers, units, all_units = self.sudoku_elements()
        digits = '123456789'  # Strings are immutable, so they are easier to use here than lists

        input_grid = self.grid  # Parse puzzle
        input_grid = {k: v for k, v in input_grid.items() if v != '0'}  # Filter out empty keys
        output_grid = {cell: digits for cell in coords}  # To start with, assume all digits 1-9 are possible

        def set_value(values, pos, val):
            """
            Eliminate all the other values except the entered val from the input position.
            The elimination function will propagate to peers and will do checks based on unit
            Return values, except return False if a contradiction is detected.
            """
            remaining_values = values[pos].replace(val, '')
            answers = []
            for v in remaining_values:
                answers.append(eliminate(values, pos, v))

            if all(answers):
                return values
            else:
                return None

        def eliminate(values, pos, val):
            """
            Eliminate val from values[pos] and propogate the elimination when possible.
            Based on two rules:
            * Any values we know immediately remove the possibility from existing in any peer.
            * When there is only possible location left in a unit, it must have the remaining value.
            """

            if val not in values[pos]:
                return values  # Already eliminated this value

            values[pos] = values[pos].replace(val, '')  # Remove value from the list

            if len(values[pos]) == 0:
                return None  # Contradiction - can't remove all the possibilities
            elif len(values[pos]) == 1:
                new_val = values[pos]  # New candidate for elimination from all peers

                # Loop over peers and eliminate this value from all of them
                for peer in peers[pos]:
                    values = eliminate(values, peer, new_val)
                    if values is None:  # Exit as soon as a contradiction is found
                        return None

            # Check for the number of remaining places the eliminated value can occupy in each unit
            for unit in units[pos]:
                possible_places = [cell for cell in unit if val in values[cell]]
                if len(possible_places) == 0:  # Contradiction - can't have no possible locations left
                    return None
                # If there is only only possible location for the eliminated digit, confirm that position
                elif len(possible_places) == 1 and len(values[possible_places[0]]) > 1:
                    if not set_value(values, possible_places[0], val):
                        return None  # Exit if the outcome is a contradiction

            return values

        # First pass, should never raise a contradiction
        # Will complete easy sudokus at this point
        for position, value in input_grid.items():
            set_value(output_grid, position, value)

        if self.validate_sudoku(output_grid):  # Finish if we're done
            return output_grid

        def guess_solution(values, depth=0):
            if values is None:
                return None  # Already failed

            if all(len(v) == 1 for k, v in values.items()):
                return values  # Solved the puzzle, can end

            # Gets the cell with the shortest length, i.e. the fewest options to try.
            # This gives the highest probability for propagating a solution correctly.
            # If there are two options, there is a 0.5 chance it is correct. If there are 5, only 0.2
            possible_values = [(len(v), k) for k, v in values.items() if len(v) > 1]
            if len(possible_values) == 0:  # Contradiction, invalid solution
                return None
            n, pos = (min([(len(v), k) for k, v in values.items() if len(v) > 1]))

            # Sort possible values for the position by the number of positions possible in peers.
            # Further increases the likelihood of making the write choice.
            # Adds ~0.01s to difficult puzzles but guarantees a fast solution for even the toughest of puzzles.
            def num_peer_possibilities(poss_val):
                return len([(cell, v) for cell, v in output_grid.items() if cell in peers[pos] and len(v) > 1 and poss_val in v])

            possible_values = ''.join(sorted(output_grid[pos], key=num_peer_possibilities))

            # Attempt all choices from our minimum choice positions.
            # It is important to run all possibilities, otherwise we hit a dead end.
            # We break as soon as it succeeds, so only one solution is found.

            # Update to the above - have now seen runaway loops that are caused by trying to solve invalid puzzles. This has
            # been mitigated by wrapping this function in a timeout handler.
            for val in possible_values:
                solution = guess_solution(set_value(values.copy(), pos, val), depth + 1)
                if solution is not None:  # Complete as soon as a valid solution is found
                    return solution

        return guess_solution(output_grid)

    
    def convert_grid(self, grid):
        #converts solved grid(=dict) to matrix so we can print it out
        solved_grid = []
        bla = []
        counter = 0
        for cell, value in grid.items():
            bla.append(int(value))
            counter += 1
            if cell[-1] == '9':
                solved_grid.append(bla)
                bla = []
                counter = 0
        return solved_grid
    
    def print_sudoku(self, grid = None):
        if grid == None:
            if(self.solved_grid == None):
                #something went wrong and the puzzle couldn't be solved
                #probably digit recognition failed
                return None
            else:
                grid = self.solved_grid
                print('Puzzle solved: ')
        else:
            grid = grid
            
        print("+" + "---+"*9)
        for i, row in enumerate(grid):
            print(("|" + " {}   {}   {} |"*3).format(*[x if x != 0 else " " for x in row]))
            if i % 3 == 2:
                print("+" + "---+"*9)
            else:
                print("+" + "   +"*9)
    def solve(self):
        #self.solved_grid = self.convert_grid(self.solve_puzzle(self.grid))
        try:
            self.solved_grid = self.convert_grid(self.solve_puzzle(self.grid))
        except:
            print("I'M SORRY DAVE.\nIT SEEMS MY VISION IS BAD, I COULDN'T FIND A SOLUTION\nI GUESS I DID NOT RECOGNISE SOME DIGITS CORRECTLY")