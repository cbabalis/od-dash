#!/usr/bin/env python
""" Module to run 4 Step Model to given arrays. """
import sys
import pandas as pd
import numpy as np

import pdb


is_A_turn = True
iterations = 1


def read_user_input(prod_cons_tn_fpath, movement_fpath, crit):
    prod_cons_tn = pd.read_csv(prod_cons_tn_fpath, delimiter='\t')
    movement = pd.read_csv(movement_fpath, delimiter='\t')
    crit_percentage = float(crit)
    return prod_cons_tn, movement, crit_percentage


def is_input_valid(prod_cons_tn, movement, crit_percentage):
    if not isinstance(prod_cons_tn, pd.DataFrame):
        return False
    if not isinstance(crit_percentage, float):
        return False
    return True


def compute_4_step_model(prod_cons_tn, movement, crit_percentage, B_j, A_i=[]):
    """ documentation here.
    """
    global iterations
    # if B_j == 1: then it is the first step. begin.
    if len(set(B_j)) == 1:
        cons = prod_cons_tn['Κατανάλωση'].tolist()
        prods = prod_cons_tn['Παραγωγές (tn)'].tolist()
        movs = movement.values.tolist()
        A_i = compute_coefficient(cons, movs, B_j)
        is_A_turn = False
        T = compute_T_i_j(A_i, prods, B_j, cons, movs)
    # if crit_percentage is satsified, then exit else
    # call again compute_4_step_model with different B_j, A_j, curr_matrix
    while not is_threshold_satisfied(T, crit_percentage, prods, cons, is_A_turn):
    #while iterations < 6:
        if is_A_turn:
            A_i = compute_coefficient(cons, movs, B_j)
            is_A_turn = False
            T = compute_T_i_j(A_i, prods, B_j, cons, movs)
        else:
            B_j = compute_coefficient(prods, movs, A_i)
            is_A_turn = True
            T =  compute_T_i_j(A_i, prods, B_j, cons, movs)
        iterations += 1
    return T


def compute_coefficient(tn, movement, existing_coef):
    coef = []
    for mv_line in movement:
        coef.append(sumprod(tn, mv_line, existing_coef))
    coef = [1/x for x in coef]
    return coef


def sumprod(li1, li2, l3):
    """ sumproduct between two lists."""
    sumproduct = 0
    for l1, l2, l3 in zip(li1, li2, l3):
        sumproduct += l1 * l2 * l3
    return sumproduct


def compute_T_i_j(A_i, prods, B_j, cons, movement):
    i_rows = []
    for ai, prod_i, mov_i in zip(A_i, prods, movement):
        j_rows = []
        for bj, con_i, m in zip(B_j, cons, mov_i):
            t = bj * con_i * m * ai * prod_i
            j_rows.append(t)
        i_rows.append(j_rows)
    return i_rows


def is_threshold_satisfied(T, threshold, prods, cons, is_A_turn):
    # compute all sums
    df = pd.DataFrame(T)
    if not is_A_turn:
        # compute cols sums
        sums = df.sum(axis=0)
        thres = compute_percentages(sums, cons, threshold, is_A_turn)
    else:
        # compute rows sums
        sums = df.sum(axis=1)
        thres = compute_percentages(sums, prods, threshold, is_A_turn)
    satisfied_thresholds = len(thres[thres['res'] < threshold])
    if satisfied_thresholds < len(thres):
        return False
    else:
        return True
    


def compute_percentages(sums, tn, threshold, is_A_turn):
    """ doc here """
    thres = pd.DataFrame([sums.tolist(), tn])
    thres = thres.T
    compute_threshold(thres)
    return thres


def compute_threshold(df):
    """this creates a new column with percentages."""
    df['res'] = (df[1] - df[0]) / df[1] * 100
    df['res'] = abs(df['res'])
    # code to check if df[1], df[0] is empty somewhere
    if df['res'].isnull().values.any():
        print("empty values are ", df['res'].isnull().sum())
        df['res'] = df['res'].fillna(0)


def test_print(a_word):
    print(a_word)


def write_matrix_to_file(T, fpath, sep='\t'):
    df = pd.DataFrame(T)
    df = df.applymap(downgrade_to_two_dec)
    df.to_csv(fpath, sep=sep)


def downgrade_to_two_dec(x):
    try:
        return '{0:.2f}'.format(x)
    except:
        return x


def read_user_input_xls(prod_cons_tn_fpath, movement_fpath, crit, sheet='for_BABIS'):
    xls = pd.ExcelFile(prod_cons_tn_fpath) #pd.read_csv(prod_cons_tn_fpath, delimiter='\t')
    prod_cons_tn = pd.read_excel(xls, sheet)
    movement = pd.read_csv(movement_fpath, delimiter='\t')
    crit_percentage = float(crit)
    return prod_cons_tn, movement, crit_percentage



def four_step_model(prod_cons_matrix_fp, antist_fp, pcntage):
    """ serious doc here. missing TODO
    """
    # read input arrays
    prod_cons_fp, mv_fp, pcnt = prod_cons_matrix_fp, antist_fp, pcntage
    prod_cons_tn, movement, crit_percentage = read_user_input(prod_cons_fp, mv_fp, pcnt)
    #test_print([prod_cons_tn, movement, crit_percentage])
    # check if arrays are ok (same length)
    assert is_input_valid(prod_cons_tn, movement, crit_percentage)
    # do some preliminary work
    B_j = [1 for i in range(0, len(prod_cons_tn))]
    T = compute_4_step_model(prod_cons_tn, movement, crit_percentage, B_j)
    print("iterations are ", iterations)
    results_file_path = 'results/output.csv'
    write_matrix_to_file(T, results_file_path, sep='\t')
    # return the path of the new matrix to show as path
    return results_file_path




def main():
    # read input arrays
    prod_cons_fp, mv_fp, pcnt = sys.argv[1], sys.argv[2], sys.argv[3]
    #prod_cons_tn, movement, crit_percentage = read_user_input(prod_cons_fp, mv_fp, pcnt)
    prod_cons_tn, movement, crit_percentage = read_user_input_xls(prod_cons_fp, mv_fp, pcnt)
    test_print([prod_cons_tn, movement, crit_percentage])
    # check if arrays are ok (same length)
    assert is_input_valid(prod_cons_tn, movement, crit_percentage)
    # do some preliminary work
    B_j = [1 for i in range(0, len(prod_cons_tn))]
    T = compute_4_step_model(prod_cons_tn, movement, crit_percentage, B_j)
    print("iterations are ", iterations)
    write_matrix_to_file(T, 'output.csv')
    # run main algorithm


if __name__ == '__main__':
    main()

