import os
import shutil
import sys

from scipy import integrate

root_dir = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(root_dir, 'nequip'))
sys.path.append(os.path.join(root_dir, 'NonlocalNNModels'))

import yaml

from NonlocalNN._keys import MEAN_DOS_KEY, MEAN_Jx_KEY
from nequip.data import AtomicData
from nequip.data.transforms import TypeMapper
from ase.io import extxyz
import numpy as np
import torch

from tqdm import tqdm

from nequip.scripts.evaluate import _load_deployed_or_traindir
from pathlib import Path

import os

def plot_vinj():
    """
    Plot the injection velocity as a function of the Fermi level. Call at the end of the file processing loop (uses
    local variables defined in the loop).
    """

    def integrate_over_E(f, energies_eV, dos_eV, fermi_level_eV, T_K=300):
        kb = 8.617e-5  # eV/K
        fermi_dirac_distribution = 1 / (1 + np.exp((energies_eV - fermi_level_eV) / (kb * T_K)))

        return integrate.trapezoid(f * dos_eV * fermi_dirac_distribution, energies_eV)

    def integrate_over_E_wo_DOS(f, energies_eV, fermi_level_eV, T_K=300):
        kb = 8.617e-5  # eV/K
        fermi_dirac_distribution = 1 / (1 + np.exp((energies_eV - fermi_level_eV) / (kb * T_K)))

        return integrate.trapezoid(f * fermi_dirac_distribution, energies_eV)

    fermi_levels = np.linspace(-0.3, 0.3, 100)

    Ns_original = []
    v_injs_original = []

    Ns = []
    Is = []
    Is_original = []
    v_injs = []

    primitive_vectors = atom.cell

    for fermi_level in fermi_levels:
        A = np.linalg.norm(np.cross(primitive_vectors[0], primitive_vectors[1])) * 1e-16  # Angstroms sq to cm sq
        N = 0.5 * integrate_over_E(f=np.ones(energies.shape), energies_eV=energies, dos_eV=pred_dos,
                                   fermi_level_eV=fermi_level)
        N_original = 0.5 * integrate_over_E(f=np.ones(true_dos.shape), energies_eV=energies,
                                            dos_eV=true_dos, fermi_level_eV=fermi_level)

        I1 = integrate_over_E_wo_DOS(f=pred_Jx, energies_eV=energies, fermi_level_eV=fermi_level)
        I1_original = integrate_over_E_wo_DOS(f=true_Jx, energies_eV=energies, fermi_level_eV=fermi_level)

        Is += [I1]
        Is_original += [I1_original]
        Ns += [N / A]
        Ns_original += [N_original / A]
        v_injs += [I1 / N]
        v_injs_original += [I1_original / N_original]

    return np.array(Ns_original), np.array(v_injs_original), np.array(Ns), np.array(v_injs), np.array(Is), np.array(Is_original)

device = 'cuda'

# Names of directories containing the models in results/from_BS_TB_combined_joint/
model_descs = ['nn-local-l1-L1-dummy-unshared-conv-Ec-BS-215-hidden-0.3-res-rescale-ldos-ln-component-mask-seed-3']
loaded_data_file = None
test_atoms = None

description = 'model_descs'

results_file = ''
if os.path.exists(results_file):
    os.remove(results_file)

for model_name in eval(description):
    negatives = {}

    model_dir = 'results/from_BS_TB_combined_joint/'
    model_path = os.path.join(model_dir, model_name)
    model, _, cutoff, type_names = _load_deployed_or_traindir(Path(model_path)/"best_model.pth", device=device)

    # Load config to identify the directory of the test data
    config_file = os.path.join(model_path, "config.yaml")
    with open(config_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    # Extract the test data directory from the config file
    train_data_file = config['dataset_file_name']
    data_dir = os.path.dirname(train_data_file)
    test_data = os.path.join(data_dir, 'combined_test_dataset.extxyz')

    if test_atoms is None:
        print(f"Loading from {test_data}")
        with open(test_data, 'r') as file:
            test_atoms = [atom for atom in tqdm(extxyz.read_extxyz(file, index=slice(0, -1)), desc='Load Test Data')]
        loaded_data_file = test_data
    else:
        assert loaded_data_file == test_data

    dos_maes = []
    Jx_maes = []
    dos_mses = []
    Jx_mses = []
    v_injs_maes = []
    v_injs_mses = []
    n_inv_maes = []
    n_inv_mses = []
    n_inv_mpes = []
    i_inv_maes = []
    i_inv_mses = []
    i_inv_mpes = []

    dos_errors = []
    jx_errors = []
    v_inj_errors = []
    n_inv_errors = []
    i_inv_errors = []

    dos_vs_e_errors = []
    jx_vs_e_errors = []

    for i, atom in enumerate(tqdm(test_atoms, desc='Testing')):
        num_layers = int(atom.info['n_layers'])
        hkl = atom.info['hkl']
        strain = atom.info['strain']
        atomic_numbers = atom.get_atomic_numbers()
        species = 'Si' if 14 in atomic_numbers else 'Ge'
        n_atoms = len(atomic_numbers)

        inference_data = AtomicData.to_AtomicDataDict(AtomicData.from_ase(atom, r_max=5.0))
        type_mapper = TypeMapper(chemical_symbol_to_type={'Si': 0, 'H': 1, 'Ge': 2})
        inference_data = type_mapper(inference_data)
        inference_data['n_layers'] = torch.tensor([1.0 * num_layers])
        inference_data['strain'] = torch.tensor([atom.info['strain']])
        inference_data['batch'] = torch.tensor([0] * n_atoms)
        inference_data['ptr'] = torch.tensor([0, 1])

        # Move all tensors to the same device as the model
        inference_data = {k: v.to(device) for k, v in inference_data.items()}

        result = model(inference_data)

        energies = atom.info['energies']
        true_dos = atom.info['dos_per_atom']
        true_Jx = atom.info['Jx_per_atom']

        pred_dos = result[MEAN_DOS_KEY].detach().cpu().numpy().flatten()
        pred_Jx = result[MEAN_Jx_KEY].detach().cpu().numpy().flatten()

        # Set values below Ec to zero
        true_dos[energies < 0.0] = 0.0
        pred_dos[energies < 0.0] = 0.0
        true_Jx[energies < 0.0] = 0.0
        pred_Jx[energies < 0.0] = 0.0

        # Injection velocity
        Ns_original, v_injs_original, Ns, v_injs, Is, Is_original = plot_vinj()

        # Calculate the MAE and MSE
        mae_dos = np.mean(np.abs(pred_dos - true_dos))
        mse_dos = np.mean((pred_dos - true_dos) ** 2)
        mae_Jx = np.mean(np.abs(pred_Jx - true_Jx))
        mse_Jx = np.mean((pred_Jx - true_Jx) ** 2)
        mae_vinj = np.mean(np.abs(v_injs - v_injs_original))
        mse_vinj = np.mean((v_injs - v_injs_original) ** 2)
        mae_ninv = np.mean(np.abs(Ns - Ns_original))
        mse_ninv = np.mean((Ns - Ns_original) ** 2)
        mpe_ninv = np.mean(np.abs(Ns - Ns_original) / np.abs(Ns_original))
        mae_i_inv = np.mean(np.abs(Is - Is_original))
        mse_i_inv = np.mean((Is - Is_original) ** 2)
        mpe_i_inv = np.mean(np.abs(Is - Is_original) / np.abs(Is_original))

        # Append errors to lists
        dos_errors.append((true_dos[energies > 0.0] - pred_dos[energies > 0.0]))
        jx_errors.append((true_Jx[energies > 0.0] - pred_Jx[energies > 0.0]))
        v_inj_errors.append((v_injs_original - v_injs))
        n_inv_errors.append((Ns_original - Ns) / Ns_original)
        i_inv_errors.append((Is_original - Is) / Is_original)

        dos_maes.append(mae_dos)
        dos_mses.append(mse_dos)
        Jx_maes.append(mae_Jx)
        Jx_mses.append(mse_Jx)
        v_injs_maes.append(mae_vinj)
        v_injs_mses.append(mse_vinj)
        n_inv_maes.append(mae_ninv)
        n_inv_mses.append(mse_ninv)
        n_inv_mpes.append(mpe_ninv)
        i_inv_maes.append(mae_i_inv)
        i_inv_mses.append(mse_i_inv)
        i_inv_mpes.append(mpe_i_inv)

        dos_vs_e_errors += [np.abs(true_dos - pred_dos)]
        jx_vs_e_errors += [np.abs(true_Jx - pred_Jx)]

    # Print the average MAE and MSE
    avg_dos_mae = np.mean(dos_maes)
    avg_dos_mse = np.mean(dos_mses)
    avg_Jx_mae = np.mean(Jx_maes)
    avg_Jx_mse = np.mean(Jx_mses)
    avg_vinj_mae = np.mean(v_injs_maes)
    avg_vinj_mse = np.mean(v_injs_mses)
    avg_ninv_mae = np.mean(n_inv_maes)
    avg_ninv_mse = np.mean(n_inv_mses)
    avg_ninv_mpe = np.mean(n_inv_mpes)
    avg_i_inv_mae = np.mean(i_inv_maes)
    avg_i_inv_mse = np.mean(i_inv_mses)
    avg_i_inv_mpe = np.mean(i_inv_mpes)

    print(f"Summary for {model_name}")
    print(f"Average MAE for DOS: {avg_dos_mae}, Average rMSE for DOS: {np.sqrt(avg_dos_mse)}")
    print(f"Average MAE for Jx: {avg_Jx_mae}, Average rMSE for Jx: {np.sqrt(avg_Jx_mse)}")
    print(f"Average MAE for v_inj: {avg_vinj_mae}, Average rMSE for v_inj: {np.sqrt(avg_vinj_mse)}")
    print(f"Average MAE for N_inv: {avg_ninv_mae}, Average rMSE for N_inv: {np.sqrt(avg_ninv_mse)}, Average absolute relative error for N_inv: {avg_ninv_mpe}")
    print(f"Average MAE for I_inv: {avg_i_inv_mae}, Average rMSE for I_inv: {np.sqrt(avg_i_inv_mse)}, Average absolute relative error for I_inv: {avg_i_inv_mpe}")

    # Print number of model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of model parameters: {num_params}")

    with open(results_file, 'a') as f:
        f.write(f"\nSummary for {model_name}\n")
        f.write(f"Average MAE for DOS: {avg_dos_mae}, Average rMSE for DOS: {np.sqrt(avg_dos_mse)}\n")
        f.write(f"Average MAE for Jx: {avg_Jx_mae}, Average rMSE for Jx: {np.sqrt(avg_Jx_mse)}\n")
        f.write(f"Average MAE for v_inj: {avg_vinj_mae}, Average rMSE for v_inj: {np.sqrt(avg_vinj_mse)}\n")
        f.write(
            f"Average MAE for N_inv: {avg_ninv_mae}, Average rMSE for N_inv: {np.sqrt(avg_ninv_mse)}, Average absolute relative error for N_inv: {avg_ninv_mpe}\n")
        f.write(f"Average MAE for I_inv: {avg_i_inv_mae}, Average rMSE for I_inv: {np.sqrt(avg_i_inv_mse)}, Average absolute relative error for I_inv: {avg_i_inv_mpe}\n")
        f.write(f"Number of model parameters: {num_params}\n")



