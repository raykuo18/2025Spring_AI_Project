# `create_env.py`

A small CLI utility to create Python virtual environments with either `conda` or the built-in `venv` module. Supports creating environments by **path** or by **name**, and optionally installing a specific Python version (for `conda`).

---

## Prerequisites

- Python ≥ 3.6  
- If using `conda` mode, [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed and on your `PATH`.

---

## Installation

```bash
# Clone your project (if not already)
git clone <project>
cd <project>/.dev

# Make script executable (UNIX) and run
chmod +x create_env.py
./create_env.py --conda --path=$(pwd)/../.env --python-version=3.11 # Recommended

# (Optional) install dev-requirements for other tools in your project
pip install -r dev-requirements.txt
```

## Usage
```
Usage: create_env.py [-h] [-v] [-c] [-p <STR>] [-py <3.11>] [-n <STR>]

Required Arguments:
  -p, --path <STR>         Absolute path to virtual environment to be created.

Optional Arguments:
  -v, --venv               Use python’s built-in venv (default if neither venv nor conda specified).
  -c, --conda              Use conda to create the environment.
  -py, --python-version    Python version to install (conda only).
  -n, --name               Name for the environment (for conda: “-n <name>”; for venv: folder `<name>`).
  -h, --help               Show this help message and exit.
```

## Examples

**NOTE: All examples assume that you are in the ``.dev`` directory.**

### ``venv`` Examples
-----

#### 1. Create a path-based venv (default)
```bash
# Creates a venv folder at ./my_env
./create_env.py --path ./my_env
```

#### 2. Create a named venv
```bash
# Creates a venv folder named "project_env" in cwd
./create_env.py --venv --name project_env --path ./unused_path
```
**Note**: ``--path`` is still required, but when ``--name`` is provided for venv, the actual folder created will be ./project_env.

### ``conda`` Examples
-----

#### 3. Create a path-based conda environment (Recommended)
```bash
# Creates a conda env in ./conda_env
./create_env.py --conda --path ./conda_env
```
**Note**: This is the recommended setup, see the example in [Installation](#installation).

#### 4. Create a named conda environment
```bash
# Creates a named conda env "analysis_env"
./create_env.py --conda --name analysis_env
```

#### 5. Create a named conda env with specific Python version
```bash
# Creates "py311_env" with Python 3.11
./create_env.py --conda --name py311_env --python-version 3.11
```

## After Environment Creation

- ``venv``

```bash
source project_env/bin/activate
```

- ``conda``
```bash
conda activate analysis_env
```

Now install dependencies:

```bash
pip install -r dev-requirements.txt
```
**Note**: See example in [Installation](#installation).