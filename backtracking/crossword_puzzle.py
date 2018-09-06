'''
Solve 10 x 10 crossword grid puzzle
Input 	   		Output

++++++++++ 		++++++++++
+------+++ 		+POLAND+++
+++-++++++ 		+++H++++++
+++-++++++ 		+++A++++++
+++-----++ 		+++SPAIN++
+++-++-+++ 		+++A++N+++
++++++-+++ 		++++++D+++
++++++-+++ 		++++++I+++
++++++-+++ 		++++++A+++
++++++++++ 		++++++++++
POLAND;LHASA;SPAIN;INDIA

Problem adapted from: Hackerrank
'''

import numpy as np
import collections
from itertools import groupby


def extract_slots(grid, num_rows=10, num_cols=10):
    """
    Extract slots from grid so that words can be put in those slots
    return: list of lists of (i,j) such that positions in same slot
    are in the same list
    """
    grid = [list(row) for row in grid]
    res = []
    # Go through rows
    for r in range(num_rows):
        for k,g in groupby(enumerate(grid[r]),key=lambda x: x[1]):
            groups = list(g)
            if groups[0][1] == '-':
                res.append([(r,i) for i,e in groups])
    # Go through columns (or rows of transpose)
    for r in range(num_cols):
        for k,g in groupby(enumerate(np.array(grid).T[r]),key=lambda x: x[1]):
            groups = list(g)
            if groups[0][1] == '-':
                res.append([(i,r) for i,e in groups])

    # Output result as dictionary with keys as length of slot
    res_dict = collections.defaultdict(list)
    for slot in res:
        res_dict[len(slot)].append(slot)
    return dict(res_dict)


def is_valid(word, slot, assignment):
    """
    Check whether assignment is valid
    Args:
        word: word to be assigned
        slot: list of grid coordinate to assign letters to
        assignment: current assignment
    Return: boolean
    """
    if len(word) != len(slot):
        raise ValueError("Word and slot do not have same length")

    for i, letter in enumerate(word):
        if (slot[i] in assignment) and assignment[slot[i]] != letter:
            return False
    return True


def assign(word, slot, assignment):
    """
    Assign word letters to slot
    """
    for i, letter in enumerate(word):
        assignment[slot[i]] = letter


def unassign(word, slot, assignment):
    """
    Unassign word letter from already assigned slot
    """
    for i, letter in enumerate(word):
        del assignment[slot[i]]


def bt(words, i, slots, assignment, indent=""):
    """
    Main backtracking function
    """
    # If all words are assigned, return True
    if i >= len(words):
        return True

    # If there are words left to be assigned, start assigning
    # to each slot, then check whether that assignment is valid
    w = words[i]
    for slot in slots[len(w)]:
        print(indent, w, slot, '::: valid =', is_valid(w, slot, assignment))
        if is_valid(w, slot, assignment):
            assign(w, slot, assignment)
            if bt(words, i+1, slots, assignment, indent+" |  "):
                return True
            unassign(w, slot, assignment)
    return False


if __name__ == '__main__':
    # Read grid and words from input
    grid = []
    with open('input.txt','r') as f:
        for i in range(10):
            grid.append(f.readline().strip())
        words = f.readline().strip().split(';')

    # Get assignment using backtracking
    slots = extract_slots(grid)
    assignment = {}
    bt(words, 0, slots, assignment)
    grid_res = [list(r) for r in grid]
    for pos, letter in assignment.items():
        i, j = pos
        grid_res[i][j] = letter

    # Write output to a file
    output = '\n'.join([''.join(r) for r in grid_res])
    print('Final output: \n{}'.format(output))
    with open('output.txt','w') as f:
        f.write(output)
    print('Output saved!')
