import numpy as np
from scipy.linalg import eigh
from time import time
import pynof
import psi4
from minimization import *

def compute_energy(mol, p=None, C=None, n=None, guess="HF", printmode=True, educ=False):
    """
    Compute the electronic energy using HF or NOF orbital rotations.

    This function evaluates the total electronic energy associated with a given
    set of molecular orbitals. When used in educational mode (educ=True), it
    also returns auxiliary information that is useful for analyzing convergence
    behavior during optimization.

    Parameters
    ----------
    mol : object
        Molecular object containing geometry and basis-set information.
    p : object
        Container for method-specific parameters (HF or NOF), including
        occupation numbers and optimization settings.
    C : numpy.ndarray, optional
        Molecular orbital coefficient matrix. If not provided, it is constructed
        internally.
    n : numpy.ndarray, optional
        Occupation numbers associated with the molecular orbitals. If None,
        an initial occupation pattern is constructed internally.
    guess : str, optional
        Initial guess for the molecular orbitals. Accepted values include
        "HF" (Hartree--Fock guess obtained with Psi4) and "Core" (diagonalization
        of the core Hamiltonian). Default is "HF".
    printmode : bool, optional
        If True, detailed information about the calculation and the optimization
        progress is printed to standard output. Default is True.
    educ : bool, optional
        If True, the function returns additional data intended for educational
        analysis (e.g., learning-rate values and energy history). Default is
        False.

    Returns
    -------
    energy : float
        Total electronic energy (including nuclear repulsion).
    alpha : float or None
        Learning rate used in the orbital optimization when `educ=True`;
        otherwise None.
    energy_history : list of tuple or None
        List of (iteration, total energy) pairs collected during the
        optimization when `educ=True`; otherwise None.

    Notes
    -----
    The function always returns the same data structure for all code paths.
    When `educ=False`, the additional educational outputs are returned as None.
    This design choice ensures a consistent and reusable API for both research
    and teaching purposes.
    """

    # TXEMA Added educ
    """Compute Natural Orbital Functional single point energy"""

    # TXEMA
    no_energy_change_count = 0
    energy_data = []
    # Txema
    t1 = time()

    wfn = p.wfn
    S, T, V, H, I, b_mnl, Dipole = pynof.compute_integrals(wfn, mol, p)

    if printmode:
        print("Number of basis functions                   (NBF)    =", p.nbf)
        if p.RI:
            print("Number of auxiliary basis functions         (NBFAUX) =", p.nbfaux)
        print("Inactive Doubly occupied orbitals up too     (NO1)    =", p.no1)
        print("No. considered Strongly Doubly occupied MOs (NDOC)   =", p.ndoc)
        print("No. considered Strongly Singly occupied MOs (NSOC)   =", p.nsoc)
        print("No. of Weakly occ. per St. Doubly occ.  MOs (NCWO)   =", p.ncwo)
        print("Dimension of the Nat. Orb. subspace         (NBF5)   =", p.nbf5)
        print("No. of electrons                                     =", p.ne)
        print("Multiplicity                                         =", p.mul)
        print("")

    # Nuclear Energy
    E_nuc = mol.nuclear_repulsion_energy()

    # Guess de MO (C)
    if C is None:
        if guess == "Core" or guess is None:
            Eguess, C = eigh(H, S)  # (HC = SCe)
        else:
            EHF, wfn_HF = psi4.energy(guess, return_wfn=True)
            EHF = EHF - E_nuc
            C = wfn_HF.Ca().np
    C = pynof.check_ortho(C, S, p)

    # Guess Occupation Numbers (n)
    if n is None:
        if p.occ_method == "Trigonometric":
            gamma = pynof.compute_gammas_trigonometric(p.ndoc, p.ncwo)
        if p.occ_method == "Softmax":
            p.nv = p.nbf5 - p.no1 - p.nsoc
            gamma = pynof.compute_gammas_softmax(p.ndoc, p.ncwo)
    else:
        if p.occ_method == "Trigonometric":
            gamma = pynof.n_to_gammas_trigonometric(n, p.no1, p.ndoc, p.ndns, p.ncwo)
        if p.occ_method == "Softmax":
            p.nv = p.nbf5 - p.no1 - p.nsoc
            gamma = pynof.n_to_gammas_softmax(n, p.no1, p.ndoc, p.ndns, p.ncwo)

    elag = np.zeros((p.nbf, p.nbf))

    E_occ, nit_occ, success_occ, gamma, n, cj12, ck12 = pynof.occoptr(
        gamma, C, H, I, b_mnl, p
    )

    E, E_old, E_diff = 9999, 9999, 9999

    if printmode:
        print("")
        if p.nof == "HF":
            print(f"HF Calculation ({p.orb_method} Optimization)")
        else:
            print(
                f"PNOF{p.ipnof} Calculation ({p.orb_method}/{p.occ_method} Optimization)"
            )
        print("==================")
        print("")
        print(
            "{:^7} {:^7}  {:^7}  {:^14} {:^14} {:^14}   {:^6}   {:^6} {:^6} {:^6}".format(
                "Nitext",
                "Nit_orb",
                "Nit_occ",
                "Eelec",
                "Etot",
                "Ediff",
                "Grad_orb",
                "Grad_occ",
                "Conv Orb",
                "Conv Occ",
            )
        )
    for i_ext in range(p.maxit):
        # orboptr
        print("alpha = ", p.alpha)
        if p.orb_method == "ADAM":
            E_orb, C, nit_orb, success_orb = orbopt_adam(gamma, C, H, I, b_mnl, p)
        if p.orb_method == "ADAM2":
            E_orb, C, nit_orb, success_orb = orbopt_adam2(gamma, C, H, I, b_mnl, p)
        if p.orb_method == "SD":
            E_orb, C, nit_orb, success_orb = orbopt_sd(gamma, C, H, I, b_mnl, p)
        if p.orb_method == "ADAGRAD":
            E_orb, C, nit_orb, success_orb = orbopt_adagrad(gamma, C, H, I, b_mnl, p)
        if p.orb_method == "RMSPROP":
            E_orb, C, nit_orb, success_orb = orbopt_rmsprop(gamma, C, H, I, b_mnl, p)
        if p.orb_method == "ADADELTA":
            E_orb, C, nit_orb, success_orb = orbopt_adadelta(gamma, C, H, I, b_mnl, p)
        if p.orb_method == "CG":
            E_orb, C, nit_orb, success_orb = orbopt_cg(gamma, C, H, I, b_mnl, p)

        # occopt
        E_occ, nit_occ, success_occ, gamma, n, cj12, ck12 = pynof.occoptr(
            gamma, C, H, I, b_mnl, p
        )
        if p.occ_method == "Softmax":
            C, gamma = pynof.order_occupations_softmax(C, gamma, H, I, b_mnl, p)

        E = E_orb
        E_diff = E - E_old
        E_old = E

        # Orbital Gradient
        y = np.zeros((p.nvar))
        grad_orb = pynof.calcorbg(y, n, cj12, ck12, C, H, I, b_mnl, p)
        # Occupation Gradient
        J_MO, K_MO, H_core = pynof.computeJKH_MO(C, H, I, b_mnl, p)
        grad_occ = pynof.calcoccg(gamma, J_MO, K_MO, H_core, p)

        print(
            f"{i_ext:6d} {nit_orb:6d} {nit_occ:6d}   {E:14.8f} {E + E_nuc:14.8f} {E_diff:15.8f}      {np.linalg.norm(grad_orb):3.1e}    {np.linalg.norm(grad_occ):3.1e}   {success_orb}   {success_occ}"
        )
        energy_data.append((i_ext, E + E_nuc))
        # TXE
        if educ:
            if abs(E_diff) < 1e-8:
                no_energy_change_count = no_energy_change_count + 1

            if no_energy_change_count > 2:
                print("")
                print("")
                print("@@@@@@@@@@@ W A R N I N G @@@@@@@@@@@@@@@@@@@")
                print(
                    "!!!! Energy is not converging, if you are playing with alpha, is not a good value "
                )
                print("!!!! In any case check orbital and occupation gradients")
                print("@@@@@@@@@@@ W A R N I N G @@@@@@@@@@@@@@@@@@@")
                print("")
                print("")
                break

        # Txema  incread convergence coherence with ADAM function
        # print(perturb, E, Estored)
        if (success_orb or np.linalg.norm(grad_orb) < 1e-4) and (
            success_occ or np.linalg.norm(grad_occ) < 1e-4
        ):
            print("--------Converged--------")
            break

    n, _ = pynof.ocupacion(
        gamma,
        p.no1,
        p.ndoc,
        p.nalpha,
        p.nv,
        p.nbf5,
        p.ndns,
        p.ncwo,
        p.HighSpin,
        p.occ_method,
    )
    cj12, ck12 = pynof.PNOFi_selector(n, p)
    E, elag, sumdiff, maxdiff = pynof.ENERGY1r(C, n, H, I, b_mnl, cj12, ck12, p)
    print(f"\nLagrage sumdiff {sumdiff:3.1e} maxfdiff {maxdiff:3.1e}")

    C, n, elag = pynof.order_subspaces(C, n, elag, H, I, b_mnl, p)

    np.save(p.title + "_C.npy", C)
    np.save(p.title + "_n.npy", n)

    if printmode:
        print("")
        print("RESULTS OF THE OCCUPATION OPTIMIZATION")
        print("========================================")

        e_val = elag[np.diag_indices(p.nbf5)]
        print(f" {"Idx":^3}    {"n":^9}   {"E (Hartree)":>12}")
        for i in range(p.nbeta):
            print(f" {i + 1:3d}    {2 * n[i]:9.7f}  {e_val[i]:12.8f}")
        for i in range(p.nbeta, p.nalpha):
            if not p.HighSpin:
                print(f" {i + 1:3d}    {2 * n[i]:9.7f}  {e_val[i]:12.8f}")
            else:
                print(f" {i + 1:3d}    {n[i]:9.7f}  {e_val[i]:12.8f}")
        for i in range(p.nalpha, p.nbf5):
            print(f" {i + 1:3d}    {2 * n[i]:9.7f}  {e_val[i]:12.8f}")

        print("")

        print("----------------")
        print(" Final Energies ")
        print("----------------")

        if guess == "HF":
            print(f"       HF Total Energy = {E_nuc + EHF:15.7f}")
        print(f"Final NOF Total Energy = {E_nuc + E:15.7f}")
        if guess == "HF":
            print(f"    Correlation Energy = {E - EHF:15.7f}")
        print("")
        print("")

    E_t = E_nuc + E

    pynof.fchk(p.title, wfn, mol, "Energy", E_t, elag, n, C, p)

    t2 = time()
    print(f"Elapsed Time: {t2 - t1:10.2f} (Seconds)")

    # TXEMA
    print(educ)
    if educ:
        alpha_out = p.alpha
        energy_data_out = energy_data
    else:
        alpha_out = None
        energy_data_out = None

    return E_t, alpha_out, energy_data_out
    


def calc_hf_orbrot(mol, p):
    """
    Compute the Hartree--Fock energy using orbital rotations.

    This function evaluates the Hartree--Fock energy by explicitly optimizing
    the energy with respect to orbital rotation parameters, rather than using
    a conventional SCF/DIIS procedure. This formulation is adopted to provide
    a controlled and transparent optimization landscape for instructional
    purposes.

    Parameters
    ----------
    mol : object
        Molecular object containing geometry and basis-set information.
    p : object
        Hartree--Fock parameter container,  including optimization algorithm and
        learning-rate settings.

    Returns
    -------
    alpha : float
        Learning rate used in the orbital optimization.
    energy_history : list of tuple
        List of (iteration, total energy) pairs collected during the optimization.
    """

    p.nof = "HF"
    p.freeze_occ = True
    n = np.zeros(p.nbf5)
    n[0 : p.ndoc] = 1

    #    pynof.compute_energy(mol,p,C=None,n=n,guess="Core",educ=True)
    _, alpha, energy_data = compute_energy(mol, p, C=None, n=n, guess="Core", educ=True)
    return alpha, energy_data


def calc_nof_orbrot(mol, p):
    """
    Compute the NOF energy using orbital rotations.

    This function evaluates the total energy within a Natural Orbital Functional
    (NOF) framework, optimizing both orbital rotations and fractional occupation
    numbers. The formulation highlights the impact of electron correlation on
    the optimization landscape.

    Parameters
    ----------
    mol : object
        Molecular object containing geometry and basis-set information.
    p : object
        NOF parameter container, including occupation numbers, optimization
        algorithm, and learning-rate settings.

    Returns
    -------
    alpha : float
        Learning rate used in the orbital optimization.
    energy_history : list of tuple
        List of (iteration, total energy) pairs collected during the optimization.
    """
    p.nof = None
    p.occ_method = "Softmax"

    _, alpha, energy_data = compute_energy(mol, p, C=None, guess="Core", educ=True)
    return alpha, energy_data
