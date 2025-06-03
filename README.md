# Network-Subsampling

This repository contains a Python-based data analysis workflow developed in Jupyter Notebook. It includes auxiliary scripts and modularized components to support reproducibility and extensibility.
Modified from: https://github.com/Ashish7129/Graph_Sampling
Original Author: Ashish Aggarwal (MIT License)


---
##  Project Structure
```
├──01_Direct_SubSampling.ipynb # Direct subsampling notebook
├──02_Combined_Subsampling.ipynb #Combined subsampling notebook
├──helper.py # Helper functions used in notebooks
├──Graph_Sampling/ # Folder containing additional Python modules
│ ├── SRW_RWF_ISRW.py
│ ├── Snowball.py
│ └── portrait_divergence.py
├── requirements.txt # List of required Python packages
└── README.md # This documentation file
```
---
##  Environment

- Python version: **3.12.2**
- Recommended: use a virtual environment (e.g., `venv` or `conda`)
- All dependencies are listed in `requirements.txt`

## How to Use
**1. Getting Started**

You can obtain the code in two ways:

Option 1: Download as ZIP

- 1.1. Go to the repository page: [https://github.com/mlizhangx/Network-Subsampling.git]
- 1.2. Click the green **"Code"** button, then select **"Download ZIP"**
- 1.3. Extract the ZIP file on your computer
- 1.4. Open the folder in your preferred editor (e.g., VSCode)

Option 2: Clone via Git (for users familiar with Git)

```bash
git clone https://github.com/mlizhangx/Network-Subsampling.git
cd yourrepo
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```
**3. Launch Jupyter and open the notebooks**

Additional Notes
- helper.py includes commonly used utility functions.
- The Graph_Sampling/ folder contains Python files for extended functionality, which can be imported in notebooks. Modified based on https://github.com/Ashish7129/Graph_Sampling.



