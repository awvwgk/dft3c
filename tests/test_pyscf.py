import pytest

from pyscf import gto
from dft3c.pyscf import dft


@pytest.fixture(params=["H2O"])
def mol(request, basis: str) -> gto.Mole:
    if request.param == "H2O":
        return gto.M(
            atom="O 0 0 0; H 0 0 1; H 1 0 0",
            charge=0,
            spin=0,
            basis=basis,
        )
    raise ValueError(f"Unsupported molecule: {request.param}")


@pytest.fixture(params=["df", "no df"])
def with_density_fit(request) -> bool:
    return request.param == "df"


@pytest.fixture(params=["r2scan/def2-svp", "r2scan-3c", "b97-3c"])
def method(request) -> str:
    return request.param


@pytest.fixture
def xc(method: str) -> str:
    if "/" in method:
        return method.split("/")[0]
    return method


@pytest.fixture
def basis(method: str) -> str:
    if "/" in method:
        return method.split("/")[-1]
    if method == "r2scan-3c":
        return "def2-mtzvpp"
    if method == "b97-3c":
        return "def2-mtzvp"
    else:
        raise ValueError(f"Unsupported method: {method}")


@pytest.fixture
def reference(method: str, with_density_fit) -> float:
    return {
        ("r2scan-3c", True): -76.41361825396947,
        ("r2scan-3c", False): -76.41358774020162,
        ("r2scan/def2-svp", True): -76.31206141715451,
        ("r2scan/def2-svp", False): -76.31197604421918,
        ("b97-3c", True): -76.3854942690059,
        ("b97-3c", False): -76.38543925321305,
    }[(method, with_density_fit)]


def test_dft(mol: gto.Mole, xc: str, with_density_fit: bool, reference: float) -> None:
    ks = dft.KS(mol, xc=xc, with_density_fit=with_density_fit)
    energy = ks.kernel()
    assert energy == pytest.approx(reference)
