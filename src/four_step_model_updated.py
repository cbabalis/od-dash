#!/usr/bin/env python
""" Module to run 4 Step Model to given arrays. """
import sys
import pandas as pd
import numpy as np
import math

import pdb


is_A_turn = True


def read_user_input(prod_cons_tn_fpath, movement_fpath, crit, group_by_col):
    prod_cons_tn = pd.read_csv(prod_cons_tn_fpath, delimiter='\t')
    prod_cons_tn = prod_cons_tn.groupby(group_by_col, as_index=False).sum()
    movement = pd.read_csv(movement_fpath, delimiter='\t')
    assert len(prod_cons_tn) == len(movement), "prod-cons and movement should be of the same length but it is %s and %s." %(len(prod_cons_tn), len(movement))
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
    # result to be returned
    final_T = []
    iterations = 0
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
    #while iterations < 7:
        if is_A_turn:
            A_i = compute_coefficient(cons, movs, B_j)
            is_A_turn = False
            T = compute_T_i_j(A_i, prods, B_j, cons, movs)
            if _has_prods_cons_integrity(T, prods, cons):
                print("it should exit now. from A. iteration is ", iterations)
                final_T = T
        else:
            B_j = compute_coefficient(prods, movs, A_i)
            is_A_turn = True
            T =  compute_T_i_j(A_i, prods, B_j, cons, movs)
            if _has_prods_cons_integrity(T, prods, cons):
                print("it should exit now. from b")
                final_T = T
        iterations += 1
        if iterations > 100 and _has_prods_cons_integrity(T, prods, cons):
            break
        print("we 're in iteration number : ", iterations)
    #return T
    return final_T


def compute_coefficient(tn, movement, existing_coef):
    coef = []
    for mv_line in movement:
        coef.append(sumprod(tn, mv_line, existing_coef))
    coef = [1/x for x in coef]
    return coef


def sumprod(li1, li2, l3):
    """ sumproduct between two lists."""
    sumproduct = 0
    try:
        for l1, l2, l3 in zip(li1, li2, l3):
            sumproduct += l1 * l2 * l3
        return sumproduct
    except TypeError:
        print("ERROR: must be numbers!")
        print("l1 is ", l1, " l2 is ", l2, "and l3 is ", l3)


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
        print("threshold is ok!")
        return True


def _has_prods_cons_integrity(T, prods, cons):
    """ method to check if sum of all rows is equal to sum of all cols.
    """
    df = pd.DataFrame(T)
    # compute total sums of all computed rows and columns
    col_sum_df = df.sum(axis=0)
    col_sum = col_sum_df.sum()
    col_sum = int(col_sum)
    
    row_sum_df = df.sum(axis=1)
    row_sum = row_sum_df.sum()
    row_sum = int(row_sum)
    
    # compute initial sums of consumptions and productions respectively
    col_sum_init = sum(cons)
    row_sum_init = sum(prods)
    if not col_sum == int(col_sum_init) or not row_sum == int(row_sum_init):
        print("ERROR!! current prods sum is ", row_sum, " and initial is ", row_sum_init)
        return False
    else:
        print("GOOD! current prods sum is ", row_sum, " and initial is ", row_sum_init)
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
    _fill_col_with_zeros(df, 'res')
    # if df['res'].isnull().values.any():
    #     print("empty values are ", df['res'].isnull().sum())
    #     df['res'] = df['res'].fillna(0)


def _fill_col_with_zeros(df, col):
    if df[col].isnull().values.any():
        print("empty values are ", df[col].isnull().sum())
        df[col] = df[col].fillna(0)


def test_print(a_word):
    print(a_word)


def write_matrix_to_file(T, fpath, sep='\t', cols=[]):
    df = pd.DataFrame(T)
    if not cols:
        print("columns are empty. problem")
    else:
        df.columns = cols
    df = df.applymap(downgrade_to_two_dec)
    df.to_csv(fpath, sep=sep)


def downgrade_to_two_dec(x):
    try:
        return '{0:.2f}'.format(x)
    except:
        return x


def populate_diagonal_elements(T, prod_cons_tn, int_mvm_col='internal_mvment'):
    """Method to populate diagonal elements of OD-matrix.
    OD matrix should have zero as diagonal elements because internal movements
    are being assigned as special case. populate diagonal elements with the
    initial quantities.

    Args:
        T (list): list of lists containing OD matrix.
        prod_cons_tn (dataframe): Dataframe with productions, consumptions and internal consumptions.
    """
    # following two lines reset index from a copy of prod-cons array for assigning
    # correct values to the final result.
    diagonal_values = prod_cons_tn[int_mvm_col].tolist()
    for (idx_i, val_i) in enumerate(T):
        for (idx_j, val_j) in enumerate(val_i):
            if idx_j == idx_i:
                val_i[idx_j] = diagonal_values.pop(0) #prod_cons_tn[int_mvm_col].pop(idx_j)


def read_user_input_xls(prod_cons_tn_fpath, movement_fpath, crit, sheet='for_BABIS'):
    xls = pd.ExcelFile(prod_cons_tn_fpath) #pd.read_csv(prod_cons_tn_fpath, delimiter='\t')
    prod_cons_tn = pd.read_excel(xls, sheet)
    movement = pd.read_csv(movement_fpath, delimiter='\t')
    crit_percentage = float(crit)
    return prod_cons_tn, movement, crit_percentage


def sort_arrays_lexicographically(df_to_sort, df_to_index, group_by_col):
    """ Method to lexicographically sort an array according to the other.
    
    This method sorts the df_to_sort dataframe's rows according to df_to_index
    dataframe's columns. A detailed example is described below.
    
    Example
    --------------------
    df_to_sort       df_to_index
    A  | B  | C  |   a2 | a3 | a1 |
    a1 | b1 | c1 |   v1 | y1 | z1 |
    a2 | b2 | c2 |   v2 | y2 | z2 |
    a3 | b3 | c3 |   v3 | y3 | z3 |
    
    The result should be:
    df_to_sort       df_to_index
    A  | B  | C  |   a2 | a3 | a1 |
    a2 | b2 | c2 |   v1 | y1 | z1 |
    a3 | b3 | c3 |   v2 | y2 | z2 |
    a2 | b2 | c2 |   v3 | y3 | z3 |

    Args:
        df_to_sort (pandas dataframe): Dataframe which values should be sorted
            according to the df_to_index columns
        df_to_index (pandas dataframe): Dataframe that provides the columns based
            on which the sorting will take place.
    """
    # create the list that will be the sorting index
    sorter = df_to_index.columns.tolist()
    sorterIndex = dict(zip(sorter, range(len(sorter))))
    # sort the column of interest values and return
    df_to_sort['sorted'] = df_to_sort[group_by_col].map(sorterIndex)
    df_to_sort = df_to_sort.sort_values(['sorted'])
    # and return the initial dataframe free from additional columns or data.
    del df_to_sort['sorted']
    return df_to_sort


def apply_internal_movement_factor(prod_cons_tn, internal_mvment_pcnt=35,
                                   prods_col='Παραγωγές (tn)', cons_col='Κατανάλωση',
                                   int_mvm_col='internal_mvment'):
    """Method to apply some values and an algorithm for balancing internal
    movements.

    Args:
        prod_cons_tn (dataframe): Productions-Consumptions matrix.
        internal_mvment_pcnt (int): Percentage for internal consumption of production.

    Returns:
        DataFrame: Balanced dataframe.
    """
    # next line is about gkassel normalization
    prod_cons_tn = balance_quantities(prod_cons_tn, prods_col, cons_col)
    # modify the internal values according to the percentage for internal consumption.
    if internal_mvment_pcnt > 0:
        print("internal movement percentage is ", internal_mvment_pcnt)
        prod_cons_tn = _set_internal_movement_column_values(prod_cons_tn, internal_mvment_pcnt,
                                                        prods_col, cons_col, int_mvm_col)
    # balance the matrix and returns
    #prod_cons_tn = balance_quantities(prod_cons_tn, prods_col, cons_col)
    return prod_cons_tn


def _set_internal_movement_column_values(df, internal_mvment_pcnt, prods_col, cons_col, int_mvm_col):
    # create the percentage
    mvment_pcntg = internal_mvment_pcnt / 100
    # create a new dataframe column with internal consumptions and
    # populate it according to the values of each row.
    df[int_mvm_col] = np.where((df[prods_col] > df[cons_col]),
                                               df[cons_col] * mvment_pcntg,
                                               df[prods_col] * mvment_pcntg)
    # remove from both consumption and production rows the rows of 
    # internal consumption, accordingly.
    df[prods_col] = df[prods_col] - df[int_mvm_col]
    df[cons_col] = df[cons_col] - df[int_mvm_col]
    return df


def balance_quantities(df, prods_col, cons_col):
    """Method to balance the total sums of quantities for the
    productions and the consumptions respectively

    Args:
        prods Dataframe): Dataframe of productions
        cons (Dataframe): Dataframe of consumptions
    """
    # sum up the productions and the consumptions respectively
    prods_sum = df[prods_col].sum()
    cons_sum = df[cons_col].sum()
    balance_factor = cons_sum / prods_sum
    # apply balance factor to the productions.
    df[prods_col] = df[prods_col] * balance_factor
    print("balance factor is ", balance_factor)
    print("production of all is ", prods_sum, " while consumption is ", cons_sum)
    return df



def four_step_model(prod_cons_matrix_fp, antist_fp, pcntage, internal_mvment_pcnt=0, group_by_col='ΠΕΡΙΦΕΡΕΙΑ', res_fpath='results/output.csv'):
    """ serious doc here. missing TODO
    """
    # read input arrays
    prod_cons_fp, mv_fp, pcnt = prod_cons_matrix_fp, antist_fp, pcntage
    prod_cons_tn, movement, crit_percentage = read_user_input(prod_cons_fp, mv_fp, pcnt, group_by_col)
    # prepare data (rebalancing, percentage in internal movements, etc)
    prod_cons_tn = apply_internal_movement_factor(prod_cons_tn, internal_mvment_pcnt)
    # check if arrays are ok (same length)
    prod_cons_tn = sort_arrays_lexicographically(prod_cons_tn, movement, group_by_col)
    assert is_input_valid(prod_cons_tn, movement, crit_percentage)
    # do some preliminary work
    B_j = [1 for i in range(0, len(prod_cons_tn))]
    T = compute_4_step_model(prod_cons_tn, movement, crit_percentage, B_j)
    if internal_mvment_pcnt > 0:
        populate_diagonal_elements(T, prod_cons_tn)
    results_file_path = res_fpath
    write_matrix_to_file(T, results_file_path, sep='\t', cols=movement.columns.tolist())
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
    write_matrix_to_file(T, 'output.csv')
    # run main algorithm


if __name__ == '__main__':
    main()

