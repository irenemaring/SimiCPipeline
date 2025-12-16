#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script calculates the AUC matrix using an object-oriented approach.
"""

import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed
from pathlib import Path
import sys


class BaseProcessor:
    """
    Base class for SimiC processors.
    Provides common functionality for file handling and data loading.
    """


    def __init__(self, p2df, p2res):
        """
        Initialize the base processor.
        
        Args:
            p2df (str): Path to dataframe pickle file
            p2res (str): Path to results pickle file
        """
        self.p2df = Path(p2df)
        self.p2res = Path(p2res)
        self.validate_files()
    
    def validate_files(self):
        """Validate that required files exist."""
        if not self.p2df.exists():
            raise FileNotFoundError(f"Data file not found: {self.p2df}")
        if not self.p2res.exists():
            raise FileNotFoundError(f"Results file not found: {self.p2res}")


class AUCProcessor(BaseProcessor):
    """
    A class to handle the AUC calculation process.
    Inherits from BaseProcessor for common functionality.
    """


    def __init__(self, p2df, p2res):
        super().__init__(p2df, p2res)
        self.normalized_weights = None
        self.original_df = None
        self.TF_ids = None
        self.target_ids = None
        self.res_dict = None
        self.adj_r2_dict = None
        self.AUC_dict = None

    def normalized_by_target_norm(self):
        """
        Load data and normalize the weight matrix with respect to target expression norms.
        Cut at the len of TFs (removed the bias term)
        """
        self.res_dict = read_pickle(self.p2res)

        weight_dic = self.res_dict['weight_dic']
        self.TF_ids = self.res_dict['TF_ids']
        self.target_ids = self.res_dict['query_targets']

        self.original_df = read_pickle(self.p2df)
        target_df = self.original_df[self.target_ids]
        
        # Euclidean norm of target expresion
        target_norms = np.linalg.norm(target_df, axis=0)

        # normalized_weights = {}
        # for label in weight_dic:
        #     normalized_mat = weight_dic[label] / target_norms
        #     normalized_weights[label] = normalized_mat[:len(TF_ids), :]

        # Remove bias term (last row) and normalize
        self.normalized_weights = {
            label: weights[:-1, :] / target_norms  # Remove last row (bias term)
            for label, weights in weight_dic.items()
        }
        self.adj_r2_dict = self.res_dict['adjusted_r_squared']

    @staticmethod
    def cal_AUC(row_vec_in, weight_vec_in, cur_adj_r2_for_all_target, sort_by, adj_r2_threshold=0.7, select_top_k_targets=None, percent_of_target=1):
        """
        Calculate the AUC score for a given row and weight vector.
        """
        if sort_by == 'expression':
            new_order = np.argsort(row_vec_in)[::-1]
        elif sort_by == 'weight':
            new_order = np.argsort(weight_vec_in)[::-1]
        elif sort_by == 'adj_r2':
            new_order = np.argsort(cur_adj_r2_for_all_target)[::-1]
        else:
            raise ValueError(f'sort_by must be one of: expression, weight, adj_r2, \n but get {sort_by}')

        ordered_weighted = weight_vec_in[new_order]
        ordered_adj_r2 = cur_adj_r2_for_all_target[new_order]

        # Filter weights by adj_r2 threshold
        ordered_weighted = ordered_weighted[ordered_adj_r2 >= adj_r2_threshold]
        
        # Debug: Check if filtering removed all weights
        if len(ordered_weighted) == 0:
            print(f"Warning: All weights filtered out by adj_r2_threshold={adj_r2_threshold}")
            print(f"  adj_r2 range: [{np.min(cur_adj_r2_for_all_target):.4f}, {np.max(cur_adj_r2_for_all_target):.4f}]")
            print(f"  Number of genes passing threshold: 0 / {len(cur_adj_r2_for_all_target)}")
            return np.nan

        len_of_genes = int(len(ordered_weighted) * percent_of_target)
        
        if len_of_genes == 0:
            print(f"Warning: len_of_genes=0 after applying percent_of_target={percent_of_target}")
            print(f"  Filtered weights length: {len(ordered_weighted)}")
            return np.nan
        
        sum_of_weight = np.sum(ordered_weighted[:len_of_genes])
        
        if select_top_k_targets is not None:
            assert isinstance(select_top_k_targets, int)
            select_top_k_targets = min(select_top_k_targets, len_of_genes)
            top_k_weight_idx = np.argpartition(ordered_weighted, -select_top_k_targets)[-select_top_k_targets:]
            ordered_weights_top_k = np.zeros_like(ordered_weighted)
            ordered_weights_top_k[top_k_weight_idx] = ordered_weighted[top_k_weight_idx]
        else:
            ordered_weights_top_k = ordered_weighted

        running_sum = np.cumsum(ordered_weights_top_k[:len_of_genes])
        AUC_score = np.sum(running_sum) / (sum_of_weight * len_of_genes)
        return AUC_score

    def _process_row(self, args):
        """
        Helper function to process a single row for parallelization.

        Args:
            args (tuple): Arguments for processing a single row.

        Returns:
            tuple: Row index and the computed AUC row.
        """

        row_idx, row, weight_mat, cur_adj_r2_for_all_target, adj_r2_threshold, select_top_k_targets, percent_of_target, sort_by = args

        tmp_AUC_row = np.zeros(len(self.TF_ids))
        row_vec_in = row[self.target_ids].values

        for tf_idx in range(len(self.TF_ids)):
            weight_vec_in = weight_mat[tf_idx, :]
            AUC_score = self.cal_AUC(
                row_vec_in, weight_vec_in, cur_adj_r2_for_all_target,
                sort_by=sort_by, adj_r2_threshold=adj_r2_threshold,
                select_top_k_targets=select_top_k_targets, percent_of_target=percent_of_target
            )
            tmp_AUC_row[tf_idx] = AUC_score

        return row_idx, tmp_AUC_row

    def get_AUCell_mat(self, adj_r2_threshold=0.7, select_top_k_targets=None, percent_of_target=1, sort_by='expression', num_cores=None):
        """
        Compute the AUC matrix for all labels and TFs using joblib for parallel processing.

        Args:
            adj_r2_threshold (float): Threshold for filtering weights by adjusted R-squared values.
            select_top_k_targets (int or None): Number of top targets to consider. If None, use all targets.
            percent_of_target (float): Fraction of targets to consider (e.g., 0.8 means 80% of targets).
            sort_by (str): Criterion for sorting targets ('expression', 'weight', or 'adj_r2').
            num_cores (int or None): Number of CPU cores to use. If None, use all available cores.
        """ 
        assert sort_by in ['expression', 'weight', 'adj_r2']

        df_in = self.original_df.reset_index(drop=True)
        original_index = self.original_df.index
        AUC_dict = {}
        # Determine the number of cores to use        
        if num_cores is None:
            num_cores = -1 # Use all available cores
        print(f"Using {num_cores if num_cores > 0 else 'all available'} cores for parallel processing.")
        if num_cores == 0:
            num_cores = 1  # Fallback to single core if 0 is provided
        for label in self.normalized_weights:
            weight_mat = np.abs(self.normalized_weights[label])
            cur_adj_r2_for_all_target = self.adj_r2_dict[label]
            
            time_start = time.time()

# Prepare arguments for parallel processing
            args = [
                (row_idx, row, weight_mat, cur_adj_r2_for_all_target, adj_r2_threshold, select_top_k_targets, percent_of_target, sort_by)
                for row_idx, row in df_in.iterrows()
            ]

# Use joblib to parallelize row processing
            try:
                results = Parallel(n_jobs=num_cores)(
                    delayed(self._process_row)(arg) for arg in args
                )
            except Exception as e:
                print(f"Joblib parallel processing failed: {e}")
                print("Falling back to sequential processing...")
                results = [self._process_row(arg) for arg in args]

# Collect results
            AUC_mat = np.zeros([df_in.shape[0], len(self.TF_ids)])
            for row_idx, tmp_AUC_row in results:
                AUC_mat[row_idx, :] = tmp_AUC_row

# End timing for the current label
            time_end = time.time()

            print(f"Label {label} processed in {time_end - time_start:.2f} seconds")
            AUC_dict[label] = pd.DataFrame(data=AUC_mat, columns=self.TF_ids, index=original_index)

        self.AUC_dict = AUC_dict
        return AUC_dict

    def save_AUC_dict(self, p2saved_file, adj_r2_threshold=0.7, select_top_k_targets=None, percent_of_target=1, sort_by='expression', num_cores=None):
        """
        Calculate and save the AUC matrix to a file.
        """
        AUC_dict = self.get_AUCell_mat(
            adj_r2_threshold=adj_r2_threshold, select_top_k_targets=select_top_k_targets,
            percent_of_target=percent_of_target, sort_by=sort_by, num_cores = num_cores
        )

        write_pickle(AUC_dict, p2saved_file)


def run_AUCprocessor(p2df, p2res, p2saved_file, percent_of_target=1, sort_by='expression', adj_r2_threshold=0.7, select_top_k_targets=None, debug=False, num_cores=0):
    """
    Main function to calculate and save the AUC matrix using the AUCProcessor class.
    """
    # Initialize the AUCProcessor
    processor = AUCProcessor(p2df, p2res)
    # Load and normalize weights
    print("Loading and normalizing weights...")
    processor.normalized_by_target_norm()
    
    # if debug:
    #     import ipdb; ipdb.set_trace()
    
    # Compute and save the AUC matrices
    print("Computing and saving AUC matrices...")
    print(f"Starting at {time.strftime('%X %x %Z')}")
    processor.save_AUC_dict(
        p2saved_file,
        adj_r2_threshold=adj_r2_threshold,
        select_top_k_targets=select_top_k_targets,
        percent_of_target=percent_of_target,
        sort_by=sort_by,
        num_cores=num_cores
    )
    print(f"AUC matrices saved to: {p2saved_file}")
    print(f"Finishing at {time.strftime('%X %x %Z')}")
