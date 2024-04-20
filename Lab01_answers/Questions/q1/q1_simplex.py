"""
Lab week 1: Question 1: Linear programs

Implement solvers to solve linear programs of the form:

max c^{T}x
subject to:
Ax <= b
x >= 0

(a) Firstly, implement simplex method covered in class from scratch to solve the LP

simplex reference:
https://www.youtube.com/watch?v=t0NkCDigq88
"""
import numpy
import pulp
import pandas as pd
import argparse


def parse_commandline_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--testDirectory', type=str, required=True, help='Directory of the test case files')
    arguments = parser.parse_args()
    return arguments


def simplex_solver(A_matrix: numpy.array, c: numpy.array, b: numpy.array) -> list:
    """
    Implement LP solver using simplex method.

    :param A_matrix: Matrix A from the standard form of LP
    :param c: Vector c from the standard form of LP
    :param b: Vector b from the standard form of LP
    :return: list of pivot values simplex method encountered in the same order
    """
    pivot_value_list = []
    ################################################################
    # %% Student Code Start
    # Implement here
    while numpy.max(c) >0 :
        a = numpy.argmax(c)
        for i in range(len(A_matrix)):
          if A_matrix[i][a]!=0:
            mini = i
        for i in range(len(A_matrix)):
          if A_matrix[i][a]!=0: 
            if  b[i]/A_matrix[i][a] <  b[mini]/A_matrix[mini][a] and b[i]/A_matrix[i][a]>0 :
               mini = i
        pivot_value_list.append(A_matrix[mini][a])
        A_matrix[mini] = A_matrix[mini]/A_matrix[mini][a]
        for i in range(len(A_matrix)):
            if i != mini :
                A_matrix[i]=A_matrix[i]- A_matrix[i][a] * A_matrix[mini]
        c = c - c[a] * A_matrix[mini]
    # %% Student Code End
    ################################################################
    

    # Transfer your pivot values to pivot_value_list variable and return
    return pivot_value_list


if __name__ == "__main__":
    # get command line args
    args = parse_commandline_args()
    if args.testDirectory is None:
        raise ValueError("No file provided")
    # Read the inputs A, b, c and run solvers
    # There are 2 test cases provided to test your code, provide appropriate command line args to test different cases.
    matrix_A = pd.read_csv("{}/A.csv".format(args.testDirectory), header=None, dtype=float).to_numpy()
    vector_c = pd.read_csv("{}/c.csv".format(args.testDirectory), header=None, dtype=float)[0].to_numpy()
    vector_b = pd.read_csv("{}/b.csv".format(args.testDirectory), header=None, dtype=float)[0].to_numpy()

    simplex_pivot_values = simplex_solver(matrix_A, vector_c, vector_b)
    for val in simplex_pivot_values:
        print(val)
