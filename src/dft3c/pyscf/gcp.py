from pyscf import gto, lib, mcscf, scf
from pyscf.grad import rhf as rhf_grad

import numpy as np
from typing import Optional, Tuple

from dftd3.interface import (
    GeometricCounterpoise,
)

GradientsBase = getattr(rhf_grad, "GradientsBase", rhf_grad.Gradients)


class GeometricCounterpoiseCorrection(lib.StreamObject):
    def __init__(
        self,
        mol: gto.Mole,
        method: str,
        basis: str,
    ):
        self.mol = mol
        self.verbose = mol.verbose
        self.method = method
        self.basis = basis

    def dump_flags(self, verbose: Optional[bool] = None):
        """
        Show options used for the GCP correction.
        """
        lib.logger.info(self, "** GCP parameter **")
        lib.logger.info(self, "method %s", self.method)
        lib.logger.info(self, "basis %s", self.basis)
        return self

    def kernel(self) -> Tuple[float, np.ndarray]:
        mol = self.mol

        lattice = None
        periodic = None
        if hasattr(mol, "lattice_vectors"):
            lattice = mol.lattice_vectors()
            periodic = np.array([True, True, True], dtype=bool)

        gcp = GeometricCounterpoise(
            np.array([gto.charge(mol.atom_symbol(ia)) for ia in range(mol.natm)]),
            mol.atom_coords(),
            lattice=lattice,
            periodic=periodic,
            method=self.method,
            basis=self.basis,
        )

        res = gcp.get_counterpoise(grad=True)

        return res.get("energy"), res.get("gradient")

    def reset(self, mol: gto.Mole):
        """Reset mol and clean up relevant attributes for scanner mode"""
        self.mol = mol
        return self


class _GCP:
    pass


class _GCPGrad:
    pass


def gcp_energy(mf: scf.hf.SCF, method: str, basis: str) -> scf.hf.SCF:

    if method is None:
        method = (
            "hf"
            if isinstance(mf, mcscf.casci.CASCI)
            else getattr(mf, "xc", "HF").upper().replace(" ", "")
        )

    if basis is None:
        basis = mf.mol.basis

    with_gcp = GeometricCounterpoiseCorrection(
        mf.mol,
        method=method,
        basis=basis,
    )

    if isinstance(mf, _GCP):
        mf.with_gcp = with_gcp
        return mf

    class GCP(_GCP, mf.__class__):
        def __init__(self, method, with_gcp):
            self.__dict__.update(method.__dict__)
            self.with_gcp = with_gcp
            self._keys.update(["with_gcp"])

        def dump_flags(self, verbose=None):
            mf.__class__.dump_flags(self, verbose)
            if self.with_gcp:
                self.with_gcp.dump_flags(verbose)
            return self

        def energy_nuc(self):
            enuc = mf.__class__.energy_nuc(self)
            if self.with_gcp:
                edisp = self.with_gcp.kernel()[0]
                mf.scf_summary["dispersion"] = edisp
                enuc += edisp
            return enuc

        def reset(self, mol=None):
            self.with_gcp.reset(mol)
            return mf.__class__.reset(self, mol)

        def nuc_grad_method(self):
            scf_grad = mf.__class__.nuc_grad_method(self)
            return gcp_grad(scf_grad)

        Gradients = lib.alias(nuc_grad_method, alias_name="Gradients")

    return GCP(mf, with_gcp)


def gcp_grad(scf_grad: GradientsBase, method: str, basis: str):
    # Ensure that the zeroth order results include DFTD3 corrections
    if not getattr(scf_grad.base, "with_gcp", None):
        scf_grad.base = gcp_energy(scf_grad.base, method=method, basis=basis)

    class GCPGrad(_GCPGrad, scf_grad.__class__):
        def grad_nuc(self, mol=None, atmlst=None):
            nuc_g = scf_grad.__class__.grad_nuc(self, mol, atmlst)
            with_gcp = getattr(self.base, "with_gcp", None)
            if with_gcp:
                gcp_g = with_gcp.kernel()[1]
                if atmlst is not None:
                    gcp_g = gcp_g[atmlst]
                nuc_g += gcp_g
            return nuc_g

    mfgrad = GCPGrad.__new__(GCPGrad)
    mfgrad.__dict__.update(scf_grad.__dict__)
    return mfgrad
