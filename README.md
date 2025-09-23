# Usage
---

## Prerequisites

- Install **Git** from [git-scm.com](https://git-scm.com/download/win).
- Install **Python 3.12** (or 3.11) from [python.org](https://www.python.org/downloads/windows/).

---

## Step 1. Clone the repository

Navigate to the folder where you want to download the code and run:

```
git clone https://github.com/LW56e2/QAOA-on-Sn-Symmetric-CSPs
cd QAOA-on-Sn-Symmetric-CSPs
```

---

## Step 2. Create and activate a virtual environment

```
python -m venv qaoa-env
qaoa-env\Scripts\activate
```

---

## Step 3. Upgrade pip

```
python -m pip install --upgrade pip
```

---

## Step 4. Install PyTorch (CPU-only build)

```
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

---

## Step 5. Install the remaining dependencies

```
python -m pip install pennylane==0.42.3 pennylane-lightning==0.42.0 numpy==1.26.4 matplotlib==3.10.5
```

---

## Step 6. Run the experiment

Inside the repo folder, run:

```
python CSP_clean.py
```

The script will create output files named like:

```
qaoa_results_YYYYMMDD_HHMMSS_xxxxxx.jsonl
```

in the results folder at
```
results/
````

---

## Notes

- No conda is required; everything uses standard Python venv + pip.
- The environment is CPU-only. GPU is not used (the script forces CPU mode).
- If you need to re-run later, just re-activate the environment with:

```
qaoa-env\Scripts\activate
```

and then run the script again.

