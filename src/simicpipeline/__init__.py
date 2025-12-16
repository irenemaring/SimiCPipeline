"""
SimiC Pipeline - Single-cell regulatory network analysis
"""

import importlib.metadata as _metadata

__version__ = _metadata.version("simicpipeline")

__all__ = ["__version__"]


try:
    from simicpipeline.core.main import SimiCPipeline
    __all__.append("SimiCPipeline")
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import SimiCPipeline: {e}")

try:
    from simicpipeline.core.aucprocessor import AUCProcessor
    __all__.append("AUProcessor")
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import AUCprocessor: {e}")

try:
    from simicpipeline.core.simicvisualization import SimiCVisualization
    __all__.append("SimiCVisualization")
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import SimiCVisualization: {e}")

try:
    from simicpipeline.core.simicpreprocess import MagicPipeline, ExperimentSetup
    __all__.extend(["MagicPipeline", "ExperimentSetup"])
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import preprocessing components: {e}")

## Extra functions from utils/io.py
try:
    from simicpipeline.utils.io import load_from_matrix_market, load_from_anndata
    __all__.extend(["load_from_matrix_market", "load_from_anndata"])
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import I/O utilities: {e}")
