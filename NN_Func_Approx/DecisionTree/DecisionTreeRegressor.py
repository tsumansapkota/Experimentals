#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = int(test_size*len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df




def coeff_of_variation(column, axis=0):
    std = np.std(column)
    mean = np.mean(column)
    cov = std/mean*100
    return cov, std, mean


def determine_type_of_feature(df):
    feature_types = []
    n_unique_values_threshold = 10
    
    for column in df.columns:
        unique_values = df[column].unique()
        example_value = unique_values[0]
        
        if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_threshold):
            feature_types.append(False)
        else:
            feature_types.append(True)
    
    return feature_types


def get_potential_splits(data, continuous):
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns-1):
        
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        continu = continuous[column_index]
        if continu:
            potential_splits[column_index] = []
            for index in range(1, len(unique_values)):
                    current_value = unique_values[index]
                    previous_value = unique_values[index-1]

                    potential_split = (current_value + previous_value)/2

                    potential_splits[column_index].append(potential_split)
        else:
            potential_splits[column_index] = unique_values
    return potential_splits



def split_data(data, split_column, split_value, continuous):
    split_column_values = data[:, split_column]
    
    conti = continuous[split_column]
    if conti:
        data_below = data[split_column_values <= split_value]
    #     data_above = data[~(split_column_values <= split_value)]
        data_above = data[split_column_values > split_value]
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    
    return data_below, data_above


def calculate_overall_SD(data_below, data_above): #standard_deviation_reduction
    bstd = np.std(data_below[:, -1])
    astd = np.std(data_above[:, -1])
    
    total_data = len(data_above) + len(data_below)
    bp = len(data_below)/total_data
    ap = len(data_above)/total_data
    
    osd = bp*bstd + ap*astd
    return osd   



def determine_best_split(data, potential_splits, continuous):
    
    overall_sd = 999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value, continuous=continuous)
            current_overall_sd = calculate_overall_SD(data_below, data_above)

            if current_overall_sd <= overall_sd:
                overall_sd = current_overall_sd
                best_split_column = column_index
                best_split_value = value
                
    return best_split_column, best_split_value #, overall_sd



class Question:

    def __init__(self, column, value, header=None,continuous=False):
        self.column = column
        self.value = value
        self.continuous = continuous
        self.header = header

    def match(self, row):
        val = row[self.column]
        if self.continuous:
            return val <= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if self.continuous:
            condition = "<="
        
        if self.header is None:
            header = self.column
        else:
            header = self.header
        return "Is %s %s %s?" % (
            header, condition, str(self.value))

class Leaf:
    def __init__(self, output):
        self.output = output
        
    def __repr__(self):
        return self.output
        
class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def regression_tree_algorithm(df, continuous=[], column_headers=None, min_samples=2, max_depth=5):
    
    # data preparations
    if column_headers is None:
        column_headers = df.columns
        continuous = determine_type_of_feature(df)
        data = df.values
    else:
        data = df
    
    # BASE_CASE

    cov, std, mean = coeff_of_variation(data[:, -1])
    if len(data)<min_samples or max_depth == 0:
        return Leaf(mean)
    
    #helper functions
    potential_splits = get_potential_splits(data, continuous)
    split_column, split_value = determine_best_split(data, potential_splits, continuous)
    data_below, data_above = split_data(data, split_column, split_value, continuous)
    if len(data_above) == 0 or len(data_below) == 0:
        return Leaf(mean)
    
    
    
    
    # RECURSIVE_PART
    else:
        max_depth -=1
             
        #instantiate sub-tree
        feature_name = column_headers[split_column]
        conti = continuous[split_column]
        question = Question(split_column, split_value, feature_name, continuous=conti)
        
        #find answers (recursion)
        yes_answer = regression_tree_algorithm(data_below,continuous, column_headers, min_samples, max_depth)
        no_answer = regression_tree_algorithm(data_above,continuous, column_headers, min_samples, max_depth)
        
        if isinstance(yes_answer, Leaf) and isinstance(no_answer, Leaf):
            if yes_answer.output == no_answer.output:
                return yes_answer

        return Decision_Node(question, yes_answer, no_answer)



def print_tree(node, spacing="    "):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing +'====='+ "Predict", node.output)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "   ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "   ")


def predict_example(example, tree):
    if isinstance(tree, Leaf):
        return tree.output
    
    if tree.question.match(example):
        return predict_example(example, tree.true_branch)
    else:
        return predict_example(example, tree.false_branch)


def calculate_error(df, tree):
    outputs = df.apply(predict_example, axis=1, args=(tree,)).values
    diff = df.output.values - outputs

    rmse = np.sqrt(np.square(diff).mean())

    return rmse