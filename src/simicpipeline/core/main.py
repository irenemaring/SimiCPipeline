#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SimiC Pipeline Class
Orchestrates the complete SimiC workflow including regression and AUC calculation.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import subprocess as sp
from pathlib import Path
from typing import Optional
from importlib.resources import files

class SimiCPipeline:
    """
    Main pipeline class for running the complete SimiC analysis workflow.
    Handles data loading, regression, filtering, and AUC calculation.
    """

    def __init__(self, 
                 project_dir: str,
                 run_name: Optional[str] = None,
                 n_tfs: int = 100,
                 n_targets: int = 1000,
                 new_run: bool = True)-> None:
        """
        Initialize the SimiC pipeline.

        Args:
            project_dir (str): Working directory for input/output
            run_name (str): Optional name for this run (used in output filenames)
            n_tfs (int): Number of transcription factors
            n_targets (int): Number of target genes
        """
        self.project_dir = Path(project_dir)
        self.run_name = run_name if run_name else "simic_run"
        self.n_tfs = n_tfs
        self.n_targets = n_targets
        
        # Initialize paths
        self.input_path = self.project_dir / "inputFiles"
        self.output_path = self.project_dir / "outputSimic"
        self.matrices_path = self.output_path / "matrices"
        
        # Create output directories if they don't exist
        self.matrices_path.mkdir(parents=True, exist_ok=True)
        
        # Default parameters
        self.lambda1 = 1e-2 # L1 regularization parameter (sparsity)
        self.lambda2 = 1e-5 # L2 regularization parameter (network similarity)e        
        self.k_cluster = None # Number of clusters of K means; only if phenotype assignment is not provided will assign labels based on them.b        
        self.similarity = True # Enables similarity constraint for Lipswitz constant (RCD process)         
        self.max_rcd_iter = 500000 # Maximum number of RCD iterations         
        self.df_with_label = False # Whether the input dataframe has labels in the last column or not. If not assignment file must be provided.         
        self.cross_val = False # Whether to perform cross-validation to select optimal lambdasf        
        self._NF = 100 # Normalization factor
        self.list_of_l1 = [1e-1, 1e-2,1e-3, 10] # List of L1 values for cross-validation
        self.list_of_l2 = [1e-1, 1e-2,1e-3, 10] # List of L2 values for cross-validationity constraint for Lipswitz constant (RCD process)
        # Timing
        self.timing = {}
        
    def set_paths(self, p2df=None, p2assignment=None, p2tf=None):
        """
        Set up all file paths for input and output.
        
        Args:
            p2df (str): Custom path to expression matrix file. Must be a pickle file with a dataframe format (genes x cells).
            p2assignment (str): Custom path to phenotype assignment file. Must be a plain text file 1 col with numbers. 0 for Basline. Should folow cell order in expression expression matrix.
            p2tf (str): Custom path to TF list file. Must be a pickle file with a list of TF names.
        """
        # Input paths - use custom paths if provided, otherwise raise error
        if p2df:
            self.p2df = Path(p2df)
        else:
            raise ValueError("Path to expression matrix dataframe (p2df) must be provided.")
            
        if p2assignment:
            self.p2assignment = Path(p2assignment)
        else:
            print("WARNING: Path to phenotype assignment file (p2assignment) not provided. Clustering will be used if k_cluster is set.")
            
        if p2tf:
            self.p2tf = Path(p2tf)
        else:
            p2tf = files("simicpipeline.data").joinpath("Mus_musculus_TF.txt")
            mouse_TF_df = pd.read_csv(p2tf,sep='\t')
            mouse_TF = mouse_TF_df['Symbol']
            with open(self.input_path / "TF_list.pickle", 'wb') as f:
                pickle.dump(mouse_TF, f)
            self.p2tf = self.input_path / "TF_list.pickle"
        
        # Output paths
        self.run_path = self.matrices_path / self.run_name
        self.run_path.mkdir(parents=True, exist_ok=True)
        base_name = f"{self.run_name}_L1_{self.lambda1}_L2_{self.lambda2}"
        self.p2simic_matrices = self.run_path / f"{base_name}_simic_matrices.pickle"
        self.p2filtered_matrices = self.run_path / f"{base_name}_simic_matrices_filtered_BIC.pickle"
        self.p2auc_raw = self.run_path / f"{base_name}_wAUC_matrices.pickle"
        self.p2auc_filtered = self.run_path / f"{base_name}_wAUC_matrices_filtered_BIC.pickle"
    
    def set_paths_custom(self, p2df=None, p2assignment=None, p2tf=None, p2simic_matrices = None, p2filtered_matrices = None, p2auc_raw = None, p2auc_filtered = None):
        """
        Set up all file paths for input and output with custom paths (intended for loading previous results).
        
        Args:
            p2df (str): Custom path to expression matrix file. Must be a pickle file with a dataframe format (genes x cells).
            p2assignment (str): Custom path to phenotype assignment file. Must be a plain text file 1 col with numbers. 0 for Basline. Should folow cell order in expression expression matrix.
            p2tf (str): Custom path to TF list file. Must be a pickle file with a list of TF names.
        """
        # Input paths - use custom paths if provided, otherwise raise error
        if p2df:
            self.p2df = Path(p2df)
        else:
            raise ValueError("Path to expression matrix dataframe (p2df) must be provided.")
            
        if p2assignment:
            self.p2assignment = Path(p2assignment)
        else:
            raise ValueError("Path to phenotype assignment file (p2assignment) not provided. Clustering will be used if k_cluster is set.")
            
        if p2tf:
            self.p2tf = Path(p2tf)
        else:
            raise ValueError("Path to TF list file (p2tf) must be provided.")
        # Output paths
        if p2simic_matrices:
            self.p2simic_matrices = Path(p2simic_matrices)
            if not self.p2simic_matrices.exists():
                raise FileNotFoundError(f"SimiC matrices file not found: {self.p2simic_matrices}")
        else:
            raise ValueError("Path to SimiC matrices file (p2simic_matrices) must be provided.")
        if p2filtered_matrices:
            self.p2filtered_matrices = Path(p2filtered_matrices)
        else:
            raise ValueError("Path to filtered matrices file (p2filtered_matrices) must be provided.")
        if p2auc_raw:
            self.p2auc_raw = Path(p2auc_raw)  
        else:
            raise ValueError("Path to AUC raw file (p2auc_raw) must be provided.")
        if p2auc_filtered: 
            self.p2auc_filtered = Path(p2auc_filtered)
        else:
            raise ValueError("Path to AUC filtered file (p2auc_filtered) must be provided.")
    
    def set_parameters(self, lambda1=None, lambda2=None, k_cluster=None, similarity=None,
                       max_rcd_iter=None, cross_val=None,list_of_l1=None,list_of_l2=None, _NF=None, run_name=None):
        """
        Set pipeline parameters.

        Args:
            lambda1 (float): L1 regularization parameter (sparsity)
            lambda2 (float): L2 regularization parameter (network similarity)
            k_cluster (int): Number of clusters of K means; only if phenotype assignment is not provided will assign labels based on them.
            similarity (bool): Enables similarity constraint for Lipswitz constant (RCD process)
            max_rcd_iter (int): Maximum RCD iterations
            cross_val (bool): Whether to perform cross-validation to select optimal lambdas
            list_of_l1 (list): List of L1 values for cross-validation
            list_of_l2 (list): List of L2 values for cross-validation
            _NF (float): Normalization factor for expression data
            run_name (str): Name for this run
        """
        if lambda1 is not None:
            self.lambda1 = lambda1
        if lambda2 is not None:
            self.lambda2 = lambda2
        if k_cluster is not None:
            self.k_cluster = k_cluster
        if similarity is not None:
            self.similarity = similarity
        if max_rcd_iter is not None:
            self.max_rcd_iter = max_rcd_iter
        if cross_val is not None:
            self.cross_val = cross_val
            self.list_of_l1 = list_of_l1
            self.list_of_l2 = list_of_l2
        if _NF is not None:
            self._NF = _NF
        if run_name is not None:
            self.run_name = run_name
        
        # Update paths based on new parameters
        self.set_paths(self.p2df, self.p2assignment, self.p2tf)

    def validate_inputs(self):
        """Validate that all required input files exist."""
        required_files = [self.p2df, self.p2assignment, self.p2tf]
        if self.p2assignment is None:
            required_files = [self.p2df, self.p2tf]
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            raise FileNotFoundError(f"Missing required input files: {missing_files}")
        
        print("✓ All required input files found")

    def run_simic_regression(self):
        """Run the SimiC LASSO regression."""
        from simicpipeline.core.clus_regression_fixed import simicLASSO_op
        
        print("\n" + "="*50)
        print(f"Running SimiC Regression")
        print(f"Run name: {self.run_name}")
        
        if self.cross_val:
            print(f"Running cross-validation with following lambdas: {self.list_of_l1} (L1), {self.list_of_l2} (L2)")
        else:
            print(f"Lambda1: {self.lambda1}, Lambda2: {self.lambda2}")
        print("="*50 + "\n")
        
        ts = time.time()
        np.random.seed(123)
        
        simicLASSO_op(
            p2df=str(self.p2df),
            p2assignment=str(self.p2assignment),
            p2tf=str(self.p2tf),
            p2saved_file=str(self.p2simic_matrices),
            similarity=self.similarity,
            k_cluster=self.k_cluster,
            num_TFs=self.n_tfs,
            num_target_genes=self.n_targets,
            _NF=self._NF,
            cross_val=self.cross_val,
            max_rcd_iter=self.max_rcd_iter,
            df_with_label=self.df_with_label,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            list_of_l1=self.list_of_l1,
            list_of_l2=self.list_of_l2
        )
        
        te = time.time()
        self.timing['simic_regression'] = te - ts
        print(f"\n✓ SimiC regression completed in {self._format_time(te - ts)}")

    def filter_weights(self, variance_threshold: float = 0.9):
        """
        Filter weights using BIC criterion.
        
        This method filters the weight matrix by keeping only the top TFs that explain
        at least `variance_threshold` of the variance for each target gene.
        
        Args:
            variance_threshold (float): Threshold for cumulative explained variance (default: 0.9)
        
        Returns:
            bool: True if filtering succeeded, False otherwise
        """
        print("\n" + "="*50)
        print("Filtering weights using BIC criterion")
        print(f"Variance threshold: {variance_threshold}")
        print("="*50 + "\n")
        
        ts = time.time()
        
        # Load the weights file
        if not self.p2simic_matrices.exists():
            print(f"Error: Weight file not found: {self.p2simic_matrices}")
            return False
        
        try:
            with open(self.p2simic_matrices, 'rb') as f:
                weights_dict = pickle.load(f)
            
            weight_dic = weights_dict['weight_dic']
            TF_ids = weights_dict['TF_ids']
            target_ids = weights_dict['query_targets']
            
            print(f"Loaded weights for {len(weight_dic)} phenotype labels")
            print(f"Number of TFs: {len(TF_ids)}")
            print(f"Number of targets: {len(target_ids)}")
            
            # Process each phenotype label
            for label in weight_dic.keys():
                print(f"\nProcessing label {label}...")
                
                # Get weight matrix for this label: (n_tfs + 1) x n_targets
                # Last row is the bias term
                weight_matrix = weight_dic[label]
                
                # Convert to DataFrame
                all_data = pd.DataFrame(
                    weight_matrix,
                    columns=target_ids
                )
                
                # Remove columns (targets) with all zero weights
                all_data = all_data.loc[:, (all_data.abs().sum(axis=0) > 0)]
                remaining_targets = all_data.columns.tolist()
                print(f"  Targets with non-zero weights: {len(remaining_targets)}/{len(target_ids)}")
                
                # Separate bias term (last row)
                bias_row = all_data.iloc[-1, :].copy()
                all_data = all_data.iloc[:-1, :]
                all_data.index = TF_ids
                
                # Scale the data by column (normalize each target by its RMS)
                # This is equivalent to R's scale(all_data, center=FALSE)
                n, p = all_data.shape 
                root_mean_sq = np.sqrt((all_data ** 2).sum(axis=0) / (n - 1))
                scaled_data = all_data / root_mean_sq
                
                max_l = []
                filtered_data = scaled_data.copy()
                
                # For each target gene
                for target in scaled_data.columns:
                    # Get absolute weights for all TFs for this target
                    target_weights = scaled_data[target].abs()
                    
                    # Sort TFs by absolute weight (descending)
                    sorted_tfs = target_weights.sort_values(ascending=False)
                    
                    # Calculate total variance for this target
                    total_variance = (scaled_data[target] ** 2).sum()
                    
                    # Find minimum number of TFs to explain variance_threshold of variance
                    cumulative_variance = 0
                    l = 0
                    
                    for tf_idx, tf in enumerate(sorted_tfs.index, 1):
                        l = tf_idx
                        # Calculate cumulative variance explained by top l TFs
                        top_l_tfs = sorted_tfs.index[:l]
                        cumulative_variance = (scaled_data.loc[top_l_tfs, target] ** 2).sum()
                        
                        # Check if we've explained enough variance
                        if total_variance > 0 and (cumulative_variance / total_variance) >= variance_threshold:
                            break
                    
                    max_l.append(l)
                    
                    # Keep only top l TFs, set others to zero
                    tfs_to_keep = sorted_tfs.index[:l]
                    tfs_to_zero = [tf for tf in scaled_data.index if tf not in tfs_to_keep]
                    filtered_data.loc[tfs_to_zero, target] = 0
                
                # Add bias term back as last row
                filtered_data = pd.concat([filtered_data, bias_row.to_frame().T])
                
                # Restore original column order (add back zero-weight targets)
                for target in target_ids:
                    if target not in filtered_data.columns:
                        filtered_data[target] = 0
                filtered_data = filtered_data[target_ids]
                
                # Convert back to numpy array
                weight_dic[label] = filtered_data.values
                
                print(f"  TFs kept per target: Mean={np.mean(max_l):.2f}, "
                      f"Median={np.median(max_l):.0f}, "
                      f"Max={np.max(max_l)}, Min={np.min(max_l)}")
            
            # Update weights_dict with filtered weights
            weights_dict['weight_dic'] = weight_dic
            
            # Save filtered weights
            with open(self.p2filtered_matrices, 'wb') as f:
                pickle.dump(weights_dict, f)
            
            te = time.time()
            self.timing['filtering'] = te - ts
            print(f"\n✓ Weight filtering completed in {self._format_time(te - ts)}")
            print(f"Filtered weights saved to: {self.p2filtered_matrices}")
            
            return True
            
        except Exception as e:
            print(f"Error during weight filtering: {e}")
            import traceback
            traceback.print_exc()
            return False

    def calculate_auc(self, use_filtered=False, adj_r2_threshold=0.7, 
                     select_top_k_targets=None, percent_of_target=1, 
                     sort_by='expression', num_cores=0):
        """
        Calculate AUC matrices.

        Args:
            use_filtered (bool): Whether to use filtered weights
            adj_r2_threshold (float): R-squared threshold for filtering
            select_top_k_targets (int): Number of top targets to select
            percent_of_target (float): Percentage of targets to consider
            sort_by (str): Sorting criterion ('expression', 'weight', 'adj_r2')
            num_cores (int): Number of cores for parallel processing
        """
        from simicpipeline.core.aucprocessor import AUCProcessor
        
        weight_file = self.p2filtered_matrices if use_filtered else self.p2simic_matrices
        output_file = self.p2auc_filtered if use_filtered else self.p2auc_raw
        
        file_type = "filtered" if use_filtered else "raw"
        print("\n" + "="*50)
        print(f"Calculating AUC matrices ({file_type} weights)")
        print("="*50 + "\n")
        
        ts = time.time()
        
        processor = AUCProcessor(str(self.p2df), str(weight_file))
        processor.normalized_by_target_norm()
        processor.save_AUC_dict(
            str(output_file),
            adj_r2_threshold=adj_r2_threshold,
            select_top_k_targets=select_top_k_targets,
            percent_of_target=percent_of_target,
            sort_by=sort_by,
            num_cores=num_cores
        )
        
        te = time.time()
        self.timing[f'auc_{file_type}'] = te - ts
        print(f"\n✓ AUC calculation completed in {self._format_time(te - ts)}")

    def run_pipeline(self, skip_filtering=False, calculate_raw_auc=False, 
                    calculate_filtered_auc=True,
                    variance_threshold=0.9,
                    auc_params=None):
        """
        Run the complete SimiC pipeline.

        Args:
            skip_filtering (bool): Skip weight filtering step. If True, it will filter out (set to 0) weights for TFs not meeting the variance threshold for that target gene.
            calculate_raw_auc (bool): Calculate AUC for raw weights
            calculate_filtered_auc (bool): Calculate AUC for filtered weights
            variance_threshold (float): Threshold for filtering (default: 0.9)
            auc_params (dict): Parameters for AUC calculation
        """
        if auc_params is None:
            auc_params = {}
        
        total_start = time.time()
        
        print("\n" + "="*70)
        print("STARTING SIMIC PIPELINE")
        print("="*70)
        
        # Validate inputs
        self.validate_inputs()
        
        # Run regression
        self.run_simic_regression()
        
        if skip_filtering and not calculate_raw_auc:
            print("\nOnly calculating simic weights! \n ✗ No filtering applied. \n ✗ No AUC calculated")
        
        # Calculate raw AUC if requested
        if calculate_raw_auc:
            self.calculate_auc(use_filtered = False, **auc_params)
        
        # Filter weights and calculate filtered AUC
        if not skip_filtering:
            filtered_success = self.filter_weights(variance_threshold=variance_threshold)
            if calculate_filtered_auc:
                if filtered_success:
                    self.calculate_auc(use_filtered=True, **auc_params)
                else:
                    print("\n✗ Skipping filtered AUC calculation due to filtering error!!")
            else:
                if filtered_success:
                    print("\n✗ Skipping filtered AUC calculation.")

        total_end = time.time()
        self.timing['total'] = total_end - total_start
        
        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print pipeline execution summary."""
        print("\n" + "="*70)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*70)
        print(f"\nRun name: {self.run_name}")
        print(f"Project directory: {self.project_dir}")
        print(f"\nParameters:")
        print(f"  - Lambda1: {self.lambda1}")
        print(f"  - Lambda2: {self.lambda2}")
        print(f"  - Number of TFs: {self.n_tfs}")
        print(f"  - Number of targets: {self.n_targets}")
        print(f"  - Clusters: {self.k_cluster}")
        print(f"\nTiming:")
        for step, duration in self.timing.items():
            print(f"  - {step}: {self._format_time(duration)}")
        print(f"\nAvailable Results:")
        for result_type in ['Ws_raw', 'Ws_filtered', 'auc_raw', 'auc_filtered']:
            try:
                self.load_results(result_type)
                print(f"✓ {result_type}")
            except FileNotFoundError:
                print(f"✗ {result_type}")
        print("\n" + "="*70)

    @staticmethod
    def _format_time(seconds):
        """Format time duration."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}min {secs}s"
        elif minutes > 0:
            return f"{minutes}min {secs}s"
        else:
            return f"{secs}s"

    def load_results(self, result_type='Ws_filtered'):
        """
        Load pipeline results.

        Args:
            result_type (str): Type of results to load ('Ws_raw', 'Ws_filtered', 'auc_raw', 'auc_filtered')

        Returns:
            dict: Loaded results
        """
        file_map = {
            'Ws_raw': self.p2simic_matrices,
            'Ws_filtered': self.p2filtered_matrices,
            'auc_raw': self.p2auc_raw,
            'auc_filtered': self.p2auc_filtered
        }
        
        if result_type not in file_map:
            raise ValueError(f"result_type must be one of {list(file_map.keys())}")
        
        result_file = file_map[result_type]
        
        if not result_file.exists():
            raise FileNotFoundError(f"Result file not found: {result_file}")
        
        with open(result_file, 'rb') as f:
            return pickle.load(f)
        
    def available_results(self):
        """
        Check which results are available.
        
        Returns:
            dict: Availability of each result type
        """
        print(f"\nAvailable Results:")
        for result_type in ['Ws_raw', 'Ws_filtered', 'auc_raw', 'auc_filtered']:
            try:
                self.load_results(result_type)
                print(f"✓ {result_type}")
            except FileNotFoundError:
                print(f"✗ {result_type}")
        print("\n" + "="*70)

# Functions for analyzing results
    def get_TF_network(self, TF_name: str, stacked: bool = False):
        """
        Retrieve the TF-target weight matrix.
        Args:
            TF_name (str): Name of the transcription factor.
            stacked (bool): If True, returns a DataFrame with targets as rows and labels as columns.
        Output:
            pd.DataFrame or dict: If stacked is True, returns a DataFrame with targets as rows and labels as columns.
                                  If stacked is False, returns a dictionary with labels as keys and Series of target weights as values.
        
        """
        simic_results = self.load_results('Ws_filtered')
        weight_dic = simic_results['weight_dic']
        TF_ids = simic_results['TF_ids']
        target_ids = simic_results['query_targets']
        if TF_name not in TF_ids:
            raise ValueError(f"TF '{TF_name}' not found in TF list.")
        print(f"Retrieving network for TF: {TF_name}")
        TF_index = TF_ids.index(TF_name)
        if stacked:
            tf_weights = [weight_dic[label][TF_index, :] for label in weight_dic.keys()]
            ws_df = pd.DataFrame(np.column_stack(tf_weights), index=target_ids, columns=weight_dic.keys())
            ws_df = ws_df.loc[(ws_df!=0).any(axis=1)]
            return ws_df

        tf_weights_filtered = {}
        for label in weight_dic.keys():
            weights = weight_dic[label]
            tf_weights = weights[TF_index, :]  # Get weights for the specified TF
            print(f"\nLabel {label}:")
            target_tupl = [(tf_weights[i], target_ids[i]) for i in range(len(target_ids)) if tf_weights[i] != 0]
            target_ws, target_names = zip(*target_tupl)
            target_ws_df = pd.Series(data=list(target_ws), index=list(target_names))
            tf_weights_filtered[label] = target_ws_df
        return tf_weights_filtered


    def analyze_weights(self):
        """
        Run weight analysis.
        Ouptut: 
            Prints Ws matrices sparsity before and after filtering.
        """
        print("\n" + "="*70)
        print("ANALYZING WEIGHT MATRICES")
        print("="*70 + "\n")
        
        # Load raw and filtered weights
        try:
            simic_results = self.load_results('Ws_raw')
            filtered_results = self.load_results('Ws_filtered')
        except FileNotFoundError as e:
            print(f"Error loading results: {e}")
            print(f"Need raw and filtered SimiC weight results to analyze weights.")
            self.available_results()
            return
        
        weight_dic_raw = simic_results['weight_dic']
        weight_dic_filtered = filtered_results['weight_dic']
        
        # Compare sparsity before and after filtering
        for label in weight_dic_raw.keys():
            weights_raw = weight_dic_raw[label]
            weights_filtered = weight_dic_filtered[label]
            
            # Remove bias term (last row) for analysis
            weights_raw_no_bias = weights_raw[:-1, :]
            weights_filtered_no_bias = weights_filtered[:-1, :] 
            # Calculate sparsity (percentage of zero weights)
            sparsity_raw = (weights_raw_no_bias == 0).sum() / weights_raw_no_bias.size * 100
            sparsity_filtered = (weights_filtered_no_bias == 0).sum() / weights_filtered_no_bias.size * 100
            
            # Calculate number of non-zero weights per target
            nonzero_per_target_raw = (weights_raw_no_bias != 0).sum(axis=0)
            nonzero_per_target_filtered = (weights_filtered_no_bias != 0).sum(axis=0)
            
            print(f"Label {label}:")
            print(f"  Raw weights:")
            print(f"    - Sparsity: {sparsity_raw:.2f}%")
            print(f"    - Avg non-zero TFs per target: {nonzero_per_target_raw.mean():.2f}")
            print(f"  Filtered weights:")
            print(f"    - Sparsity: {sparsity_filtered:.2f}%")
            print(f"    - Avg non-zero TFs per target: {nonzero_per_target_filtered.mean():.2f}")
            print(f"  Reduction: {((sparsity_filtered - sparsity_raw) / (100 - sparsity_raw) * 100):.2f}% more sparse")
            print()

    def subset_label_specific_auc(self, result_type, label):
        """
        Extract from auc results the label specific AUC dataframe.
        Args:
            result_type (str): Type of AUC results to load ('auc_raw' or 'auc_filtered')
            label (str or int): Phenotype label to subset
        Output:
            pd.DataFrame: AUC dataframe for the specified label
        """    
        auc_dic = self.load_results(result_type)
        auc_df = auc_dic[label]
        # Load cell assignments to get which cells belong to which label
        assignment_df = pd.read_csv(self.p2assignment, sep='\t', header=None, names=['label'])
        # Get cells that belong to this label
        cells_idx_in_label = assignment_df[assignment_df['label'] == int(label)].index.to_list()
        # Subset AUC dataframe to only cells in this label
        auc_subset = auc_df.iloc[cells_idx_in_label,]
        return auc_subset

    def analyze_auc_scores(self):
        """
        Run AUC score analysis.
        Ouptut: 
            Prints basic statistics and top TFs by average activity scores.
        """
        print("\n" + "="*70)
        print("ANALYZING AUC SCORES")
        print("="*70 + "\n")
        
        # Load AUC results
        try:
            auc_filtered = self.load_results('auc_filtered')
        except FileNotFoundError as e:
            print(f"Error loading results: {e}")
            print(f"Need auc_filtered results to analyze AUC scores.")
            self.available_results()
            return
        
        # Analyze each label
        for label in auc_filtered.keys():
            # Subset AUC dataframe to only cells in this label
            auc_subset = self.subset_label_specific_auc('auc_filtered',label)
            
            print(f"Label {label}:")
            print(f"  Shape: {auc_subset.shape} (cells x TFs)")
            print(f"  AUC score statistics:")
            print(f"    - Mean: {np.nanmean(auc_subset.values):.4f}")
            print(f"    - Median: {np.nanmedian(auc_subset.values):.4f}")
            print(f"    - Std: {np.nanstd(auc_subset.values):.4f}")
            print(f"    - Min: {np.nanmin(auc_subset.values):.4f}")
            print(f"    - Max: {np.nanmax(auc_subset.values):.4f}")
            
            # Find top 5 TFs with highest average AUC
            mean_auc_per_tf = auc_subset.mean(axis=0)
            top_tfs = mean_auc_per_tf.nlargest(5)
            print(f"  Top 5 TFs by average AUC:")
            for tf, score in top_tfs.items():
                print(f"    - {tf}: {score:.4f}")
            print()

    def calculate_dissimilarity(self, select_labels=None, verbose=True):
        """
        Compare AUC scores between different labels calculating dissimilarity score (0 = similar distributions, higher = more dissimilarity)
        Args:
            select_labels (list): List of labels to compare. If None, compare all available labels
        
        Output: 
            pd.DataFrame with TFs and their dissimilarity scores (sorted in descending order).
        """
        if verbose:
            print("\n" + "="*70)
            print("CALCULATING DISSIMILARITY SCORES ACROSS LABELS")
            print("="*70 + "\n")
        try:
            auc_filtered = self.load_results('auc_filtered')
        except FileNotFoundError as e:
            print(f"Error loading results: {e}")
            print(f"Need auc_filtered results to compare labels.")
            self.available_results()
            return
        
        if select_labels:
            print(f"Comparing labels {select_labels}.")
            labels = select_labels
        else:
            labels = list(auc_filtered.keys())
        if len(labels) < 2:
            print("Only one label, cannot compare!")
            return
        
        auc_dic = {}
        for label in labels:
            auc_dic[label] = self.subset_label_specific_auc('auc_filtered', label)
        if verbose:
            print(f"\nCalculating dissimilarity scores...")
        n_breaks = 100
        MinMax_val = []
        tf_names = auc_dic[labels[0]].columns.tolist()
        
        for tf in tf_names:
            # Get AUC values for this TF from both labels
            Wauc_dist = {}
            for label in labels:
                Wauc_dist[label] = np.histogram(auc_dic[label][tf].dropna(), bins=np.linspace(0, 1, n_breaks + 1), density=True)[0]
            # Create matrix of distributions
            # Extract the .values from each DataFrame (2D numpy arrays)
            arrays = [df for df in Wauc_dist.values()]
            mat = np.vstack(arrays)
            # Remove columns with all NaN
            mat = mat[:, ~np.isnan(mat).all(axis=0)]
            
            if mat.shape[1] > 0:
                # Calculate minmax difference
                minmax_diff = np.nanmax(mat, axis=0) - np.nanmin(mat, axis=0)
                variant = np.sum(np.abs(minmax_diff)) / n_breaks
                
                # Normalize by number of non-zero rows
                non_zero_rows = np.sum(np.sum(mat, axis=1) != 0)
                if non_zero_rows > 0:
                    variant = variant / non_zero_rows
            else:
                variant = 0.0
            
            MinMax_val.append(variant)
        
        # Create DataFrame with dissimilarity scores
        MinMax_df = pd.DataFrame({
            'TF': tf_names,
            'MinMax_score': MinMax_val
        }).set_index('TF')
        
        # Sort by dissimilarity score
        MinMax_df_sorted = MinMax_df.sort_values('MinMax_score', ascending=False)
        if verbose:
            print(f"\nTop 10 TFs by MinMax dissimilarity score:")
            for tf, row in MinMax_df_sorted.head(10).iterrows():
                print(f"  {tf}: {row['MinMax_score']:.4f}")
        
        return MinMax_df_sorted
    