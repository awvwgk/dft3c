from pyscf import gto, lib, mcscf, scf
from pyscf.grad import rhf as rhf_grad

import numpy as np
from typing import Dict, Optional, Tuple

from dftd3.interface import (
    DispersionModel,
    RationalDampingParam,
    ZeroDampingParam,
    ModifiedRationalDampingParam,
    ModifiedZeroDampingParam,
    OptimizedPowerDampingParam,
    CSODampingParam,
)

GradientsBase = getattr(rhf_grad, "GradientsBase", rhf_grad.Gradients)

_damping_param = {
    "d3bj": RationalDampingParam,
    "d3zero": ZeroDampingParam,
    "d3bjm": ModifiedRationalDampingParam,
    "d3mbj": ModifiedRationalDampingParam,
    "d3zerom": ModifiedZeroDampingParam,
    "d3mzero": ModifiedZeroDampingParam,
    "d3op": OptimizedPowerDampingParam,
    "d3cso": CSODampingParam,
}


class DFTD3Dispersion(lib.StreamObject):
    def __init__(
        self,
        mol: gto.Mole,
        xc: str = "hf",
        version: str = "d3bj",
        atm: bool = False,
        param: Optional[Dict[str, float]] = None,
    ):
        self.mol = mol
        self.verbose = mol.verbose
        self.xc = xc
        self.param = param
        self.atm = atm
        self.version = version

    def dump_flags(self, verbose: Optional[bool] = None):
        """
        Show options used for the DFT-D3 dispersion correction.
        """
        lib.logger.info(self, "** DFTD3 parameter **")
        lib.logger.info(self, "func %s", self.xc)
        lib.logger.info(
            self, "version %s", self.version + "-atm" if self.atm else self.version
        )
        return self

    def kernel(self) -> Tuple[float, np.ndarray]:
        """
        Compute the DFT-D3 dispersion correction.

        The dispersion model as well as the parameters are created locally and
        not part of the state of the instance.

        Returns
        -------
        float, ndarray
            The energy and gradient of the DFT-D3 dispersion correction.

        Examples
        --------
        >>> from pyscf import gto
        >>> import dftd3.pyscf as disp
        >>> mol = gto.M(
        ...     atom='''
        ...          Br    0.000000    0.000000    1.919978
        ...          Br    0.000000    0.000000   -0.367147
        ...          N     0.000000    0.000000   -3.235006
        ...          C     0.000000    0.000000   -4.376626
        ...          H     0.000000    0.000000   -5.444276
        ...          '''
        ... )
        >>> d3 = disp.DFTD3Dispersion(mol, xc="PBE0")
        >>> energy, gradient = d3.kernel()
        >>> energy
        array(-0.00303589)
        >>> gradient
        array([[ 0.00000000e+00,  0.00000000e+00,  9.66197638e-05],
               [ 0.00000000e+00,  0.00000000e+00,  2.36000434e-04],
               [ 0.00000000e+00,  0.00000000e+00, -1.16718302e-04],
               [ 0.00000000e+00,  0.00000000e+00, -1.84332770e-04],
               [ 0.00000000e+00,  0.00000000e+00, -3.15691249e-05]])
        """
        mol = self.mol

        lattice = None
        periodic = None
        if hasattr(mol, "lattice_vectors"):
            lattice = mol.lattice_vectors()
            periodic = np.array([True, True, True], dtype=bool)

        disp = DispersionModel(
            np.array([gto.charge(mol.atom_symbol(ia)) for ia in range(mol.natm)]),
            mol.atom_coords(),
            lattice=lattice,
            periodic=periodic,
        )

        if self.param is not None:
            param = _damping_param[self.version](**self.param)
        else:
            param = _damping_param[self.version](
                method=self.xc,
                atm=self.atm,
            )

        res = disp.get_dispersion(param=param, grad=True)

        return res.get("energy"), res.get("gradient")

    def reset(self, mol: gto.Mole):
        """Reset mol and clean up relevant attributes for scanner mode"""
        self.mol = mol
        return self


class _DFTD3:
    """
    Stub class used to identify instances of the `DFTD3` class
    """

    pass


class _DFTD3Grad:
    """
    Stub class used to identify instances of the `DFTD3Grad` class
    """

    pass


def d3_energy(mf: scf.hf.SCF, method: str | None = None, **kwargs) -> scf.hf.SCF:
    if method is None:
        method = (
            "hf"
            if isinstance(mf, mcscf.casci.CASCI)
            else getattr(mf, "xc", "HF").upper().replace(" ", "")
        )

    with_dftd3 = DFTD3Dispersion(
        mf.mol,
        xc=method,
        **kwargs,
    )

    if isinstance(mf, _DFTD3):
        mf.with_dftd3 = with_dftd3
        return mf

    class DFTD3(_DFTD3, mf.__class__):
        def __init__(self, method, with_dftd3):
            self.__dict__.update(method.__dict__)
            self.with_dftd3 = with_dftd3
            self._keys.update(["with_dftd3"])

        def dump_flags(self, verbose=None):
            mf.__class__.dump_flags(self, verbose)
            if self.with_dftd3:
                self.with_dftd3.dump_flags(verbose)
            return self

        def energy_nuc(self):
            enuc = mf.__class__.energy_nuc(self)
            if self.with_dftd3:
                edisp = self.with_dftd3.kernel()[0]
                mf.scf_summary["dispersion"] = edisp
                enuc += edisp
            return enuc

        def reset(self, mol=None):
            self.with_dftd3.reset(mol)
            return mf.__class__.reset(self, mol)

        def nuc_grad_method(self):
            scf_grad = mf.__class__.nuc_grad_method(self)
            return d3_grad(scf_grad)

        Gradients = lib.alias(nuc_grad_method, alias_name="Gradients")

    return DFTD3(mf, with_dftd3)


def d3_grad(scf_grad: GradientsBase, **kwargs):
    # Ensure that the zeroth order results include DFTD3 corrections
    if not getattr(scf_grad.base, "with_dftd3", None):
        scf_grad.base = d3_energy(scf_grad.base, **kwargs)

    class DFTD3Grad(_DFTD3Grad, scf_grad.__class__):
        def grad_nuc(self, mol=None, atmlst=None):
            nuc_g = scf_grad.__class__.grad_nuc(self, mol, atmlst)
            with_dftd3 = getattr(self.base, "with_dftd3", None)
            if with_dftd3:
                disp_g = with_dftd3.kernel()[1]
                if atmlst is not None:
                    disp_g = disp_g[atmlst]
                nuc_g += disp_g
            return nuc_g

    mfgrad = DFTD3Grad.__new__(DFTD3Grad)
    mfgrad.__dict__.update(scf_grad.__dict__)
    return mfgrad
