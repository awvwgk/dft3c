"""Top-level package for dft3c.

This module applies a small compatibility shim for the PySCF density-fit
gradient path used by the ASE calculator. Newer NumPy releases return a
slightly different contraction descriptor from ``numpy.einsum_path`` than the
version expected by the bundled PySCF helper, which causes the ASE gradient
path to fail with a ``ValueError``/``AttributeError`` during the density-fit
J-matrix build.
"""

from __future__ import annotations

import pyscf.lib as _pyscf_lib
from pyscf.lib import numpy_helper as _numpy_helper


def _compat_einsum(subscripts: str, *tensors, **kwargs):
    """Compatibility wrapper for PySCF's einsum helper.

    NumPy 2.x returns contraction tuples with a different layout than the
    older PySCF helper expects. This wrapper accepts both layouts and falls
    back to the internal helpers used by PySCF itself.
    """

    subscripts = subscripts.replace(" ", "")
    if len(tensors) <= 1 or "..." in subscripts:
        return _numpy_helper._numpy_einsum(subscripts, *tensors, **kwargs)
    if len(tensors) <= 2:
        return _numpy_helper._contract(subscripts, *tensors, **kwargs)

    optimize = kwargs.pop("optimize", True)
    tensors = list(tensors)
    contraction_list = _numpy_helper._einsum_path(
        subscripts, *tensors, optimize=optimize, einsum_call=True
    )[1]

    out = None
    for contraction in contraction_list:
        if len(contraction) >= 4:
            inds, _, einsum_str, _ = contraction[:4]
        else:
            inds, einsum_str, _ = contraction[:3]

        tmp_operands = [tensors.pop(x) for x in inds]
        out = (
            _numpy_helper._numpy_einsum(einsum_str, *tmp_operands)
            if len(tmp_operands) > 2
            else _numpy_helper._contract(einsum_str, *tmp_operands)
        )
        tensors.append(out)

    return out


# Patch both the top-level PySCF helper and the internal numpy_helper module.
_numpy_helper.einsum = _compat_einsum
_pyscf_lib.einsum = _compat_einsum
