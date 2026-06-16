"""
PySCF integration for DFT-3c methods.
"""

from typing import Any, Union

from pyscf import dft as pyscf_dft
from pyscf import gto

from dft3c.pyscf.d3 import d3_energy
from dft3c.pyscf.d4 import d4_energy
from dft3c.pyscf.gcp import gcp_energy
from dft3c.pyscf.util import DFT3cConfig


def KS(
    mol: gto.Mole,
    xc: str,
    *,
    with_density_fit: bool = False,
    with_newton: bool = False,
    auxbasis: str | None = None,
    ks_config: dict[str, Any] | None = None,
    soscf_config: dict[str, Any] | None = None,
) -> Union[pyscf_dft.rks.RKS, pyscf_dft.uks.UKS]:
    if mol.spin == 0:
        return RKS(
            mol,
            xc,
            with_density_fit=with_density_fit,
            with_newton=with_newton,
            auxbasis=auxbasis,
            ks_config=ks_config,
            soscf_config=soscf_config,
        )
    else:
        return UKS(
            mol,
            xc,
            with_density_fit=with_density_fit,
            with_newton=with_newton,
            auxbasis=auxbasis,
            ks_config=ks_config,
            soscf_config=soscf_config,
        )


def RKS(
    mol: gto.Mole,
    xc: str,
    *,
    with_density_fit: bool = False,
    with_newton: bool = False,
    auxbasis: str | None = None,
    ks_config: dict[str, Any] | None = None,
    soscf_config: dict[str, Any] | None = None,
) -> pyscf_dft.rks.RKS:
    config = DFT3cConfig.from_xc(xc)

    ks = pyscf_dft.RKS(mol, xc=config.xc)

    return _apply_ks_config(
        ks,
        with_density_fit=with_density_fit,
        with_newton=with_newton,
        config=config,
        auxbasis=auxbasis,
        ks_config=ks_config,
        soscf_config=soscf_config,
    )


def UKS(
    mol: gto.Mole,
    xc: str,
    *,
    with_density_fit: bool = False,
    with_newton: bool = False,
    auxbasis: str | None = None,
    ks_config: dict[str, Any] | None = None,
    soscf_config: dict[str, Any] | None = None,
) -> pyscf_dft.uks.UKS:
    config = DFT3cConfig.from_xc(xc)
    ks = pyscf_dft.UKS(mol, xc=config.xc)

    return _apply_ks_config(
        ks,
        with_density_fit=with_density_fit,
        with_newton=with_newton,
        config=config,
        auxbasis=auxbasis,
        ks_config=ks_config,
        soscf_config=soscf_config,
    )


def _apply_ks_config(
    ks: Union[pyscf_dft.rks.RKS, pyscf_dft.uks.UKS],
    *,
    with_density_fit: bool,
    with_newton: bool,
    config: DFT3cConfig,
    auxbasis: str | None,
    ks_config: dict[str, Any] | None,
    soscf_config: dict[str, Any] | None,
) -> Union[pyscf_dft.RKS, pyscf_dft.UKS]:
    """Apply common KS configuration (grids, density fitting, Newton, SOSCF)."""
    if ks_config is not None:
        ks = ks(**ks_config)

    if config.with_dftd3:
        ks = d3_energy(ks, method=config.d3_method)
    if config.with_dftd4:
        ks = d4_energy(ks, method=config.d4_method)
    if config.with_gcp:
        ks = gcp_energy(ks, method=config.gcp_method, basis=ks.mol.basis)
    if with_density_fit:
        ks = ks.density_fit(auxbasis=auxbasis)
    elif auxbasis is not None:
        raise ValueError(
            "Auxiliary basis can only be set when density fitting is enabled."
        )
    if with_newton:
        ks = ks.newton()
        if soscf_config is not None:
            ks.__dict__.update(soscf_config)
    return ks
