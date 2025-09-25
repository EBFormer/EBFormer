# EBFormer

## Prerequisites
```bash
git clone https://github.com/EBFormer/EBFormer.git
cd EBFormer
python3 -m venv .venv           
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset (EBFormer 2025)

The training / evaluation data used in the EBFormer NeurIPS 2025 submission is hosted on Zenodo:
[**Download Dataset on Zenodo**](https://zenodo.org/records/17197896?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjgwNTRiOWM0LTk1ZTItNDZkNi04YzFkLTk3OWExNjcxNDYwYyIsImRhdGEiOnt9LCJyYW5kb20iOiIwNzMyZDAzYzQxZGViMDgzOTdmY2EwNWFmMTBkNzA2NyJ9.5afkHSH6Vnf3Up98xW8lH6MNgCuMGqRfWWPiPVohU0e_iHdSDRqMh4rQhPhGb8Ve3YkXTGaU7pa123Dgxv-1_g)

*The Zenodo page also includes a information regarding dataset format and structure. 
Once downloaded, place the extracted directory inside `NonlocalNNModels/Data/`.*

---

## Directory Layout
```
nequip (local version)
NonlocalNNModels/
├── configs/            # YAML config files
├── Data/               # your datasets (see below)
└── NonlocalNN/
    ├── scripts/
    │   ├── train.py
    │   └── test.py
    └── …               # model code
```
---

## Training a Model
1. **Navigate** to the project root (`NonlocalNNModels`).
2. **Run** the training script with your chosen YAML configuration:
   ```bash
   python3 NonlocalNN/scripts/train.py configs/<desired_config>.yaml --warn-unused
   ```
   > **Configs**  
   > All example configs—including those used in the *EBFormer* NeurIPS 2025 submission—are to be placed in `configs/`.  

---

## Testing / Inference
1. **Locate** your checkpoint folder (e.g., `results/...`).
2. **Edit** `NonlocalNN/scripts/test.py` to include the directory in the specified field
3. **Run**
   ```bash
   python3 NonlocalNN/scripts/test.py
   ```
---
