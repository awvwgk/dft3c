import pytest

try:
    import ase
    from ase import Atoms
    from ase.build import molecule
    from dft3c.ase import DFT3c
except ModuleNotFoundError:
    ase = None


@pytest.fixture(params=["H2O"])
def atoms(request) -> "Atoms":
    return molecule(request.param)


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
def reference(method: str, with_density_fit: bool) -> float:
    return {
        ("r2scan-3c", False): -2079.4941395457540,
        ("r2scan-3c", True): -2079.4950466047535,
        ("r2scan/def2-svp", False): -2076.7019605622136,
        ("r2scan/def2-svp", True): -2076.7043903581562,
        ("b97-3c", False): -2078.715130379277,
        ("b97-3c", True): -2078.7167546130217,
    }[(method, with_density_fit)]


@pytest.mark.skipif(ase is None, reason="ASE not installed")
def test_ase_calculator(
    atoms: "Atoms", xc: str, basis: str, with_density_fit: bool, reference: float
) -> None:
    atoms.calc = DFT3c(xc=xc, basis=basis, with_density_fit=with_density_fit)
    energy = atoms.get_potential_energy()
    assert energy == pytest.approx(reference)
