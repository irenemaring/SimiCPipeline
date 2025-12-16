from __future__ import annotations

from typing import Optional, Tuple, Union
from pathlib import Path

import pandas as pd

# Install packages

def install_package(package_name):
   """Install a package using pip3."""
   import subprocess
   import sys
   subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def load_from_anndata(
    path: Union[str, Path],
) -> object:
    """
    Load an AnnData file (.h5ad) and return a DataFrame (cells x genes) with optional obs/var metadata.

    Args:
        path: Path to the .h5ad file.
        cells_index_name: Column name for cells index when returned as DataFrame.

    Returns:
        adata: AnnData object loaded from the file.
    """
    try:
        import anndata as ad  # type: ignore
    except Exception as e:
        raise ImportError("anndata is required to load .h5ad files") from e

    path = Path(path)
    adata = ad.read_h5ad(str(path))
    return adata

def load_from_matrix_market(
    matrix_path: Union[str, Path],
    genes_path: Union[str, Path] = None,
    cells_path: Union[str, Path] = None,
    transpose: bool = False,
    cells_index_name: str = "Cell",
) -> object:
    """
    Load a Matrix Market file (.mtx) and genes/cells TSV and return a pandas DataFrame.

    Args:
        matrix_path: Path to matrix.mtx(.gz).
        genes_path: Optional path to genes/features list (TSV with first column as names).
        cells_path: Optional path to cells/barcodes list (TSV with first column as names).
        transpose: If True, transpose the matrix (use when mtx is genes x cells).
        cells_index_name: Index name for cells rows.

    Returns:
        df: Expression matrix DataFrame (cells as rows, genes as columns).
    """
    try: 
        import scipy
    except Exception as e:
        raise ImportError("scipy is required to load Matrix Market files") from e

    matrix_path = Path(matrix_path)

    mtx = scipy.io.mmread(str(matrix_path))
    
    if scipy.sparse.issparse(mtx):
        mat = mtx.tocsr()
    else:
        mat = scipy.sparse.csr_matrix(mtx)

    if transpose:
        mat = mat.T
    # Load names
    genes = None
    cells = None
    if genes_path and genes_path.exists():
        genes = pd.read_csv(genes_path, header=None, sep="\t").iloc[:, 0].astype(str).tolist()
    if cells_path and cells_path.exists():
        cells = pd.read_csv(cells_path, header=None, sep="\t").iloc[:, 0].astype(str).tolist()

    # Convert to dense for simplicity; adjust if large matrices are expected
    arr = mat.toarray()
    df = pd.DataFrame(arr)

    # Assign indices/columns when provided
    if cells is not None and len(cells) == df.shape[0]:
        df.index = cells
    else:
        df.index = [f"cell_{i}" for i in range(df.shape[0])]
    df.index.name = cells_index_name

    if genes is not None and len(genes) == df.shape[1]:
        df.columns = genes
    else:
        df.columns = [f"gene_{j}" for j in range(df.shape[1])]

    return df