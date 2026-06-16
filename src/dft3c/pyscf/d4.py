from pyscf import gto, lib
from pyscf.grad import rhf as rhf_grad

from typing import Tuple

import numpy as np

from dftd4.interface import DampingParam, DispersionModel

GradientsBase = getattr(rhf_grad, "GradientsBase", rhf_grad.Gradients)


class DFTD4Dispersion(lib.StreamObject):
    def __init__(
        self,
        mol,
        xc: str = "hf",
        atm: bool = True,
        model: str = "d4",
    ):
        self.mol = mol
        self.verbose = mol.verbose
        self.xc = xc
        self.atm = atm
        self.model = model
        self.edisp = None
        self.grads = None

    def dump_flags(self, verbose=None) -> "DFTD4Dispersion":
        """
        Show options used for the DFT-D4 dispersion correction.
        """
        lib.logger.info(self, "** DFTD4 parameter **")
        lib.logger.info(self, "func %s", self.xc)
        return self

    def kernel(self) -> Tuple[float, np.ndarray]:
        """
        Compute the DFT-D4 dispersion correction.

        The dispersion model as well as the parameters are created locally and
        not part of the state of the instance.

        Returns
        -------
        float, ndarray
            The energy and gradient of the DFT-D4 dispersion correction.
        """
        mol = self.mol

        lattice = None
        periodic = None
        if hasattr(mol, "lattice_vectors"):
            lattice = mol.lattice_vectors()
            periodic = np.array([True, True, True], dtype=bool)

        kwargs = {}
        if self.xc == "r2scan-3c":
            kwargs["ga"] = 2.0
            kwargs["gc"] = 1.0

        disp = DispersionModel(
            np.asarray([gto.charge(sym) for sym in mol.elements]),
            mol.atom_coords(),
            mol.charge,
            lattice=lattice,
            periodic=periodic,
            model=self.model,
            **kwargs,
        )

        param = DampingParam(
            method=self.xc,
            atm=self.atm,
        )

        res = disp.get_dispersion(param=param, grad=True)

        self.edisp = res.get("energy")
        self.grads = res.get("gradient")
        return self.edisp, self.grads

    def reset(self, mol) -> "DFTD4Dispersion":
        """
        Reset mol and clean up relevant attributes for scanner mode
        """
        self.mol = mol
        return self


class _DFTD4:
    pass


class _DFTD4Grad:
    pass


def d4_energy(mf, method: str | None = None, model: str = "d4"):
    from pyscf.mcscf import casci

    if method is None:
        method = (
            "hf"
            if isinstance(mf, casci.CASCI)
            else getattr(mf, "xc", "HF").upper().replace(" ", "")
        )

    with_dftd4 = DFTD4Dispersion(
        mf.mol,
        xc=method,
        model=model,
    )

    if isinstance(mf, _DFTD4):
        mf.with_dftd4 = with_dftd4
        return mf

    class DFTD4(_DFTD4, mf.__class__):
        """
        Patched SCF class including DFT-D4 corrections.
        """

        def __init__(self, method, with_dftd4: DFTD4Dispersion):
            self.__dict__.update(method.__dict__)
            self.with_dftd4 = with_dftd4
            self._keys.update(["with_dftd4"])

        def dump_flags(self, verbose=None) -> "DFTD4":
            mf.__class__.dump_flags(self, verbose)
            if self.with_dftd4:
                self.with_dftd4.dump_flags(verbose)
            return self

        def energy_nuc(self) -> float:
            enuc = mf.__class__.energy_nuc(self)
            if self.with_dftd4:
                edisp = self.with_dftd4.kernel()[0]
                self.scf_summary["dispersion"] = edisp
                enuc += edisp
            return enuc

        def reset(self, mol=None) -> "DFTD4":
            self.with_dftd4.reset(mol)
            return mf.__class__.reset(self, mol)

        def nuc_grad_method(self):
            scf_grad = mf.__class__.nuc_grad_method(self)
            return d4_grad(scf_grad)

        Gradients = lib.alias(nuc_grad_method, alias_name="Gradients")

    return DFTD4(mf, with_dftd4)


def d4_grad(mfgrad: GradientsBase):
    # Ensure that the zeroth order results include DFTD4 corrections
    if not getattr(mfgrad.base, "with_dftd4", None):
        mfgrad.base = d4_energy(mfgrad.base)

    class DFTD4Grad(_DFTD4Grad, mfgrad.__class__):
        """
        Patched SCF class including DFT-D4 corrections.
        """

        def grad_nuc(self, mol=None, atmlst=None):
            nuc_g = mfgrad.__class__.grad_nuc(self, mol, atmlst)
            with_dftd4 = getattr(self.base, "with_dftd4", None)
            if with_dftd4:
                disp_g = with_dftd4.kernel()[1]
                if atmlst is not None:
                    disp_g = disp_g[atmlst]
                nuc_g += disp_g
            return nuc_g

    dgrad = DFTD4Grad.__new__(DFTD4Grad)
    dgrad.__dict__.update(mfgrad.__dict__)
    return dgrad
