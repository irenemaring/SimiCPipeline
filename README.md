# SimiCPipeline

A Python package for runnin SimiC a single-cell gene regulatory network inference method that jointly infers distinct, but related, gene regulatory dynamics per phenotype class. 

## Prerequisites

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

### Installing Poetry

**Linux, macOS, Windows (WSL):**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Windows (PowerShell):**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

After installation, add Poetry to your PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"  # Linux/macOS
```

Verify installation:
```bash
poetry --version
```

For more installation options, see the [official Poetry documentation](https://python-poetry.org/docs/#installation).

## Installation

### Standard Installation

Clone the repository and install using Poetry:

```bash
git clone https://github.com/irenemaring/SimiCPipeline.git
cd SimiCPipeline
poetry install
```

### Development Installation

To install with development dependencies (testing, linting, etc.):

```bash
poetry install --with dev
```

### With Documentation Tools

```bash
poetry install --with dev,docs
```

## Usage

### Activating the Poetry Environment

```bash
poetry run python
```

### Running Python Code

```python
import simicpipeline 
from simicpipeline import MagicPipeline, ExperimentSetup
from simicpipeline import SimiCPipeline, AUCprocessor
from simicpipeline import SimiCVisualization

print(f"SimiCPipeline version: {simicpipeline.__version__}")
```

### Example: MAGIC Preprocessing

```python
import anndata as ad
from simicpipeline import MagicPipeline

# Load your data
adata = ad.read_h5ad("path/to/your/data.h5ad")

# Initialize pipeline
pipeline = MagicPipeline(
    input_data=adata,
    project_dir="./my_project",
    filtered=False
)

# Run preprocessing steps
pipeline.filter_cells_and_genes(
    min_cells_per_gene=10,
    min_umis_per_cell=500
).normalize_data().run_magic(
    t=3,
    knn=5,
    save_data=True
)
```

### Example: SimiC Analysis

```python
from simicpipeline.core.simicpipeline import SimiCPipeline

# Initialize pipeline
simic = SimiCPipeline(
    workdir="./my_project",
    run_name="experiment_1",
    n_tfs=100,
    n_targets=1000
)

# Set input paths
simic.set_paths(
    p2df="./my_project/magic_output/magic_data.pickle",
    p2assignment="./my_project/inputFiles/phenotype_assignment.txt",
    p2tf="./data/Mus_musculus_TF.txt"
)

# Set parameters
simic.set_parameters(
    lambda1=1e-2,
    lambda2=1e-5,
    similarity=True,
    max_rcd_iter=500000
)

# Run complete pipeline
simic.run_pipeline(
    skip_filtering=False,
    calculate_raw_auc=False,
    calculate_filtered_auc=True
)
```

## Poetry Commands Reference

### Managing Dependencies

```bash
# Add a new dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name

# Update dependencies
poetry update

# Show installed packages
poetry show

# Show dependency tree
poetry show --tree
```

### Running Commands

```bash
# Run a script without activating shell
poetry run python script.py

# Run pytest
poetry run pytest

# Run with specific Python version
poetry env use python3.10
```

### Deploying jupyter notebook kernel
```bash
# Create a kernel using poetry .venv
poetry run python -m ipykernel install --user --name simicpipeline --display-name "Python (SimiCPipeline)"
```

### Building and Publishing

```bash
# Build distribution packages
poetry build

# Publish to PyPI
poetry publish

# Build and publish
poetry publish --build
```

## Development

### Running Tests

```bash
poetry run pytest

# With coverage
poetry run pytest --cov=simicpipeline tests/
```

### Code Formatting

```bash
# Check code for issues
poetry run ruff check src/ tests/

# Check and automatically fix issues
poetry run ruff check --fix src/ tests/

# Format code
poetry run ruff format src/ tests/

# Check formatting without making changes
poetry run ruff format --check src/ tests/
```

## Project Structure

```
SimiCPipeline/
├── src/
│   └── simicpipeline/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── simicpreprocess.py
│       │   ├── simicpipeline.py
│       │   └── simicvisualization.py
│       └── utils/
│           └── _helper_functions.py
├── tests/
├── data/
├── pyproject.toml
├── README.md
└── LICENSE
```

## Requirements

- Python >=3.10
- Poetry >=1.0.0

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use SimiCPipeline in your research, please cite:

```bibtex
@software{simicpipeline,
  author = {Marín-Goñi, Irene},
  title = {SimiCPipeline: A Python Package for SimiC Analysis},
  year = {2025},
  url = {https://github.com/irenemaring/SimiCPipeline}
}
```

## Contact

Irene Marín-Goñi - imarin.4@alumni.unav.es

Project Link: [https://github.com/irenemaring/SimiCPipeline](https://github.com/irenemaring/SimiCPipeline)