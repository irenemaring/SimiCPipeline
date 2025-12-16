#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SimiC Preprocessing Pipeline
Handles data loading, MAGIC imputation, gene selection, and file preparation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Union
import pickle
# Single-cell data manipulation
# import scipy
# import scprep
try:
    import anndata as ad
except ImportError:
    raise ImportError("Anndata is required. Please install.")
# MAGIC imputation
# import magic


class SimiCPreprocess:
    """
    Base preprocessing pipeline for SimiC analysis.
    Provides common utilities for both MAGIC imputation and experiment setup.
    """

    def __init__(self, project_dir: Union[str, Path]):
        """
        Initialize base preprocessing pipeline.

        Args:
            project_dir: Directory for project files
        """
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)


class MagicPipeline(SimiCPreprocess):
    """
    MAGIC imputation pipeline.  
    Handles loading raw expression data, filtering, normalization, and MAGIC imputation.  
    """
            
    def __init__(self, 
                 input_data: Union[ad.AnnData, pd.DataFrame],
                 project_dir: Union[str, Path],
                 magic_output_file: str = 'magic_data_allcells_sqrt.pickle',
                 filtered: bool = False):
        """
        Initialize MAGIC pipeline.  

        Args:
            project_dir:   Directory for output files
            magic_output_file:  Filename for MAGIC-imputed matrix (saved in project_dir)
            filtered: Flag to indicate if the data is filtered
        """
        super().__init__(project_dir)
        # Initialize data containers
        if isinstance(input_data, pd.DataFrame):
            # Convert DataFrame to AnnData
            
            obs=pd.DataFrame(index=input_data.index)
            var=pd.DataFrame(index=input_data.columns)
            # Convert to sparse if not already
            try:
                import scipy
                if not scipy.sparse.issparse(input_data.values):
                    matx = scipy.sparse.csr_matrix(input_data.values)
                else:
                    matx = input_data.values
            except:
                print("Warning: scipy is not installed, cannot convert to sparse matrix.")
                matx = input_data.values
            self.adata = ad.AnnData(
                    X=matx,
                    obs=obs,
                    var=var)
        else:
            self.adata = input_data
            

        # Create MAGIC output directory
        self.magic_output_dir = self.project_dir / 'magic_output'
        self.magic_output_dir.mkdir(exist_ok=True)
        
        # Set output file path
        self.magic_output_file = self.magic_output_dir / magic_output_file
        
        self.magic_adata = None
        
        # Track pipeline state
        self._filtered = filtered
        self._imputed = False

    def __repr__(self) -> str:
        """
        String representation showing pipeline state and data dimensions.
        
        Returns:
            Formatted string with pipeline information
        """
        lines = ["MagicPipeline("]
        
        # Data status
        if self.adata is None:
            lines.append("  data=None,")
        else:
            lines.append(f"  data=AnnData object with n_obs × n_vars = {self.adata.shape[0]} × {self.adata.shape[1]},")
        
        # Filtered status
        lines.append(f"  filtered={self._filtered},")
        
        # Imputed status
        lines.append(f"  imputed={self._imputed},")
        
        # MAGIC data dimensions if available
        if self.magic_adata is not None:
            lines.append(f"  magic_data=AnnData object with n_obs × n_vars = {self.magic_adata.shape[0]} × {self.magic_adata.shape[1]},")
        else:
            lines.append("  magic_data=None,")
        
        # Output directory
        lines.append(f"  project_dir='{self.project_dir}'")
        
        lines.append(")")
        
        return "\n".join(lines)
    
    def filter_cells_and_genes(self,
                               min_cells_per_gene: int = 10,
                               min_umis_per_cell: int = 500) -> 'MagicPipeline':
        """
        Filter cells and genes based on expression thresholds.
        
        Note: The original logic filters genes by number of cells expressing them,
        and cells by total UMI counts.

        Args:
            min_cells_per_gene: Minimum number of cells that must express a gene (gene filtering)
            min_umis_per_cell: Minimum total UMIs per cell (cell filtering)
            
        Returns:
            Self for method chaining
        """
        if self.adata is None:
            raise ValueError("No data loaded. Please load data first using load_from_matrix_market() or load_from_anndata().")
        
        if self._filtered:
            print("Warning: Data has already been filtered. Filtering again!")
        
        print("\nFiltering cells and genes...")
        print(f"Before filtering: {self.adata.shape[0]} cells x {self.adata.shape[1]} genes")
        
        # GENES x CELLS
        X = self.adata.X

        # 1. Filter genes - keep genes expressed in more than min_cells_per_gene cells
        bool_mat = X > 0
        cells_per_gene = np.array(bool_mat.sum(axis=0)).reshape(-1)
        keep_genes = cells_per_gene > min_cells_per_gene
        if sum(keep_genes) == 0:
            raise ValueError("All genes filtered out!  Consider lowering min_cells_per_gene.")
        
        assert keep_genes.shape[0] == X.shape[1], "Gene mask shape mismatch"
        print(f"Keeping {sum(keep_genes)}/{len(keep_genes)} genes")
        
        X = X[:, keep_genes]
        
        # 2. Filter cells - keep cells with more than min_umis_per_cell UMIs
        umis_per_cell = np.array(X.sum(axis=1)).reshape(-1)
        keep_cells = umis_per_cell > min_umis_per_cell
        if sum(keep_cells) == 0:
            raise ValueError("All cells filtered out! Consider lowering min_umis_per_cell.")
        
        assert keep_cells.shape[0] == X.shape[0], "Cell mask shape mismatch"
        print(f"Keeping {sum(keep_cells)}/{len(keep_cells)} cells")
        
        if sum(keep_cells) == len(keep_cells):
            print("All cells pass the filter!")
        
        X = X[keep_cells, :]
        # Update AnnData object
        self.adata = ad.AnnData(
            X=X,
            obs=self.adata.obs.iloc[keep_cells].copy(),
            var=self.adata.var.iloc[keep_genes].copy()
        )
        
        print(f"After filtering: {self.adata.shape[0]} cells x {self.adata.shape[1]} genes")
        
        self._filtered = True
        self._imputed = False  # Reset imputation flag since data changed

        return self
    
    def normalize_data(self) -> 'MagicPipeline':
        """
        Normalize data using library size normalization and square root transformation.

        Returns:
            Self for method chaining
        """
        if self.adata is None:
            raise ValueError("No data loaded. Please load data first using load_from_matrix_market() or load_from_anndata().")
        
        print("\nNormalizing data...")
        try:
            import scprep
        except ImportError:
            raise ImportError("scprep is required for normalization. Please install or normalize data externally.")        
        
        # Normalize the matrix. Make sure is cells x genes matrix because scprep expects [n_samples, n_features]
        matrix_sparse_norm = scprep.normalize.library_size_normalize(self.adata.X)
        sqrt_mat = np.sqrt(matrix_sparse_norm)
        
        # Update AnnData object
        self.adata = ad.AnnData(
            X=sqrt_mat,
            obs=self.adata.obs.copy(),
            var=self.adata.var.copy()
        )
        
        print(f"After normalization: {self.adata.shape[0]} cells x {self.adata.shape[1]} genes")
        
        self._imputed = False  # Reset imputation flag since data changed
        return self

    def run_magic(self,
                  save_data: bool = True,
                  n_jobs: Optional[int] = -2,
                   **kwargs) -> 'MagicPipeline':
        """
        Run MAGIC imputation on the data.

        Args:
            t: Number of diffusion steps
            knn: Number of nearest neighbors
            decay: Decay rate for kernel
            sqrt_transform: Whether to apply square root transformation
            n_jobs: Number of jobs for parallel processing
            **kwargs:  Additional arguments to pass to magic. MAGIC()
            
        Returns:
            Self for method chaining
        """
        if self.adata is None:
            raise ValueError("No data loaded. Please load data first using load_from_matrix_market() or load_from_anndata().")
        
        if not self._filtered:
            print("Warning: Data has not been filtered. Consider running filter_cells_and_genes() first.")
        
        print("\nRunning MAGIC imputation...")
        
        try:
            import magic
        except ImportError:
            raise ImportError("MAGIC is not installed. Please install it with: pip install magic-impute")
        
        magic_op = magic.MAGIC(n_jobs=n_jobs, **kwargs)
        magic_result = magic_op.fit_transform(self.adata.X)
        
        # Create new AnnData with imputed values
        self.magic_adata = ad.AnnData(
            X=magic_result,
            obs=self.adata.obs.copy(),
            var=self.adata.var.copy()
        )
        
        self._imputed = True
        print(f"MAGIC imputation complete:  {self.magic_adata.shape[0]} cells x {self.magic_adata.shape[1]} genes")
        if save_data:
            self.save_magic_data(self.magic_output_file)
        return self
    
    def save_magic_data(self, filepath: Optional[Union[str, Path]] = None) -> 'MagicPipeline':
        """
        Save MAGIC-imputed data to file.

        Args:
            filepath: Optional custom path.  If None, uses self.magic_output_file
            
        Returns:
            Self for method chaining
        """
        if self.magic_adata is None: 
            raise ValueError("No MAGIC-imputed data available. Please run run_magic() first.")
        
        output_file = Path(filepath) if filepath else self.magic_output_file
        # Check extension
        if output_file.suffix not in ['.pickle', '.h5ad']:
            raise ValueError(f"Unsupported file format: {output_file.suffix}. Use .pickle or .h5ad")
        # Check that output_path file exists and print warning
        if output_file.exists():
            print(f"Warning: Output file {output_file} already exists and will be overwritten.")
        
        print(f"\nSaving MAGIC-imputed data to {output_file}...")
        
        # Save based on file extension
        if output_file.suffix == '.pickle':
            with open(output_file, 'wb') as f:
                pickle.dump(self.magic_adata, f)
        elif output_file.suffix == '.h5ad':
            self.magic_adata.write_h5ad(output_file)
        else:
            raise ValueError(f"Unsupported file format: {output_file.suffix}. Use .pickle or .h5ad")
        
        print(f"Saved successfully to {output_file}")
    
    def is_filtered(self) -> bool:
        """Check if data has been filtered."""
        return self._filtered
    
    def is_imputed(self) -> bool:
        """Check if MAGIC imputation has been run."""
        return self._imputed
    
    def get_pipeline_status(self) -> dict:
        """
        Get current state of the pipeline. 
        
        Returns:
            Dictionary with pipeline state information
        """
        return {
            'data_loaded': self.adata is not None,
            'filtered':  self._filtered,
            'imputed': self._imputed,
            'n_cells': self.adata.shape[0] if self.adata is not None else 0,
            'n_genes': self.adata.shape[1] if self.adata is not None else 0,
            'magic_data_available': self.magic_adata is not None
        }


class ExperimentSetup(SimiCPreprocess):
    """
    Minimal experiment setup for SimiC analysis.

    Accepts input expression data (AnnData or pandas DataFrame), converts to NumPy,
    and provides:
      - calculate_mad_genes: compute MAD and select top genes
      - save_experiment_files_csv: save matrix, gene names, and cell names as CSV files
    """

    def __init__(self,
                 input_data: Union[ad.AnnData, pd.DataFrame],
                 tf_path: Union[str, Path],
                 project_dir: Union[str, Path]):
        """
        Initialize ExperimentSetup.

        Args:
            input_data: AnnData (cells x genes) or pandas DataFrame (cells x genes)
            tf_path: Path to transcription factor list file
            project_dir: Directory for output files
        """
        super().__init__(project_dir)

        # Load TF list
        self.tf_list = self._load_tf_list(tf_path)
        
        # Create standard SimiC directory structure
        self._create_directory_structure()

        # Normalize input into NumPy matrix with cell/gene names
        if isinstance(input_data, ad.AnnData):
            X = input_data.X
            # Convert sparse to dense if needed
            if hasattr(X, "toarray"):
                X = X.toarray()
            self.matrix = np.asarray(X)  # cells x genes
            self.cell_names = input_data.obs_names.tolist()
            self.gene_names = input_data.var_names.tolist()
        elif isinstance(input_data, pd.DataFrame):
            # Expect DataFrame as cells x genes
            self.matrix = input_data.values
            self.cell_names = input_data.index.astype(str).tolist()
            self.gene_names = input_data.columns.astype(str).tolist()
        else:
            raise TypeError("input_data must be an AnnData or a pandas DataFrame")

        # Basic validation
        if self.matrix.ndim != 2:
            raise ValueError("Expression matrix must be 2-dimensional (cells x genes)")
        if len(self.cell_names) != self.matrix.shape[0]:
            raise ValueError("Cell names size does not match matrix rows")
        if len(self.gene_names) != self.matrix.shape[1]:
            raise ValueError("Gene names size does not match matrix columns")

    def _load_tf_list(self, tf_path: Union[str, Path]) -> List[str]:
        """Load transcription factor list from file."""
        tf_path = Path(tf_path)
        if not tf_path.exists():
            raise FileNotFoundError(f"TF list file not found: {tf_path}")
        if tf_path.suffix == '.csv':
            tf_df = pd.read_csv(tf_path, header=None, names=["TF"])
        else:
            tf_df = pd.read_csv(tf_path, header=None, names=["TF"], sep='\t')
        return tf_df['TF'].tolist()

    def _create_directory_structure(self) -> None:
        """Create standard SimiC directory structure."""
        # Input files directory
        self.input_files_dir = self.project_dir / 'inputFiles'
        self.input_files_dir.mkdir(parents=True, exist_ok=True)
        
        # Output SimiC directories
        self.output_simic_dir = self.project_dir / 'outputSimic'
        self.figures_dir = self.output_simic_dir / 'figures'
        self.matrices_dir = self.output_simic_dir / 'matrices'
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.matrices_dir.mkdir(parents=True, exist_ok=True)

    def calculate_mad_genes(self,
                            n_tfs: int,
                            n_targets: int) -> tuple[List[str], List[str]]:
        """
        Calculate Median Absolute Deviation and select top TF and target genes.

        Args:
            n_tfs: Number of top TF genes to select based on MAD
            n_targets: Number of top target genes to select based on MAD

        Returns:
            Tuple of (final_TF_list, final_TARGET_list)
        """
        # Create expression matrix as genes x cells
        expr_matrix = pd.DataFrame(
            self.matrix.T,  # Transpose to genes x cells
            index=self.gene_names,
            columns=self.cell_names
        )
        
        # MAD calculation
        # Calculation of Median Absolute Deviation (MAD)
        mymean = expr_matrix.mean(axis=1)  # Mean expression of each gene (i) across all cells (j)
        dev_mat = expr_matrix.to_numpy() - mymean[:, None]  # Deviation matrix Xij - mean(Xi)
        mymad = np.median(np.abs(dev_mat), axis=1)  # Median of the absolute deviation of each gene (i) across all cells (j)
        MAD = pd.Series(mymad, index=expr_matrix.index)  # Create a pandas series with gene names as index
        
        # Select top MAD TF genes
        TFsmedian = MAD.reindex(set(self.tf_list)).dropna()
        TFsmedian.sort_values(ascending=False, inplace=True)  # Sort the series in descending order
        
        if len(TFsmedian) < n_tfs:
            n_tfs = min(n_tfs, len(TFsmedian))
            if n_tfs == 0:
                raise ValueError("n_tfs must be at least 1.")
            print(f"Only {len(TFsmedian)} TFs found in dataset. Selecting top {n_tfs} TFs based on MAD.")
        
        final_TF_list = TFsmedian.head(n_tfs).index.tolist()
        
        # Select the top MAD target genes
        target_set = set(expr_matrix.index).difference(set(self.tf_list))  # Get target genes
        TARGETs = MAD[target_set]  # No need to reindex as MAD is generated from expr_matrix
        
        print("Removing " + str(sum(TARGETs == 0)) + " targets with MAD = 0")  # remove TARGETs with MAD = 0
        TARGETs = TARGETs[TARGETs > 0]
        TARGETs.sort_values(ascending=False, inplace=True)  # Sort the series in descending order
        
        n_targets = min(n_targets, len(TARGETs))
        print(f"Selecting top {n_targets} targets based on MAD.")
        final_TARGET_list = TARGETs.head(n_targets).index.tolist()  # Top target genes
        
        return final_TF_list, final_TARGET_list

    def save_experiment_files(self,
                              run_data: Union[ad.AnnData, pd.DataFrame],
                              matrix_filename: str = 'expression_matrix.pickle',
                              tf_filename: str = 'TF_list.pickle',
                              annotation: Optional[str] = None) -> None:
        """
        Save matrix (cells x genes) and TF names to pickle files in inputFiles directory.

        Args:
            run_data: AnnData or pandas DataFrame (cells x genes) to save
            matrix_filename: Filename for the expression matrix pickle format (saved in project_dir/inputFiles/)
            tf_filename: Filename for the TF names in pickle format (saved in project_dir/inputFiles/)
            annotation: Optional annotation column name in run_data.obs to save as txt file
        """
        # Build paths in inputFiles directory
        matrix_filename = Path(matrix_filename)
        if matrix_filename.suffix != '.pickle':
            matrix_filename = matrix_filename.with_suffix('.pickle')
            print(f"Warning: Matrix filename must have a .pickle suffix. Changing to {matrix_filename}")
        matrix_path = self.input_files_dir / matrix_filename
        if matrix_path.exists():
            print(f"Warning: Output file {matrix_path} already exists and will be overwritten.")
        
        tf_filename = Path(tf_filename)
        if tf_filename.suffix != '.pickle':
            tf_filename = tf_filename.with_suffix('.pickle')
            print(f"Warning: TF filename must have a .pickle suffix. Changing to {tf_filename}")
        tf_path = self.input_files_dir / tf_filename
        
        # Generate df from run_data
        if isinstance(run_data, ad.AnnData):
            X = run_data.X
            obs = run_data.obs_names.tolist()
            var = run_data.var_names.tolist()
            # Convert sparse to dense if needed
            if hasattr(X, "toarray"):
                X = X.toarray()
            df = pd.DataFrame(
                X,
                index=obs,
                columns=var
            )
        else:
            df = run_data
        
        # Save matrix in pickle format
        with open(matrix_path, 'wb') as f:
            pickle.dump(df, f)
            print(f"Saved expression matrix to {matrix_path}")

        # Save TF list
        # Check TFs found in final dataset
        data_tfs = list(set(self.tf_list).intersection(set(df.columns)))
        if len(data_tfs) == 0:
            raise ValueError("No TFs found in expression matrix from the provided TF list.")
        if tf_path.exists():
            print(f"Warning: Output file {tf_path} already exists and will be overwritten.")
        with open(tf_path, 'wb') as f:
            pickle.dump(data_tfs, f)
            print(f"Saved {len(data_tfs)} TFs to {tf_path}")
        
        if annotation and isinstance(run_data, ad.AnnData):
            if annotation in run_data.obs.columns:
                print(f"Annotation '{annotation}' found in obs columns.")
                # Check annotation column is numeric
                annot_series = run_data.obs[annotation]
                if not pd.api.types.is_numeric_dtype(annot_series):
                    print("Warning: annotation is not numeric. Will convert from categorical to numeric.")
                    annot_series = pd.Series(pd.factorize(annot_series)[0], index=annot_series.index)
                try:
                    import collections
                    counter = collections.Counter(annot_series)
                    print(f"Annotation distribution: {dict(counter)}")
                except:
                    print(f"Annotation values: {set(annot_series)}")
                # Save annotation
                annot_path = self.input_files_dir / f"{annotation}_annotation.txt"
                if annot_path.exists():
                    print(f"Warning: Output file {annot_path} already exists and will be overwritten.")
                annot_series.to_csv(annot_path, index=False, header=False)
                print(f"Saved annotation to {annot_path}")
            else:
                print(f"Warning: Annotation '{annotation}' not found in obs columns.")
                print(f"Available columns: {list(run_data.obs.columns)}")
                print(f"Please manually provide an appropriate annotation file to SimiCPipeline in {self.input_files_dir}")
        elif annotation:
            print(f"Warning: Cannot save annotation. run_data must be AnnData object.")
            print(f"Please manually provide an appropriate annotation file to SimiCPipeline in {self.input_files_dir}")
        
        print("Experiment files saved successfully.")

