from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class DFT3cConfig:
    xc: str
    d3_method: str | None
    d4_method: str | None
    gcp_method: str | None
    with_dftd3: bool
    with_dftd4: bool
    with_gcp: bool

    @classmethod
    def from_xc(cls, xc: str) -> "DFT3cConfig":
        with_gcp, gcp_method = False, None
        with_dftd3, d3_method = False, None
        with_dftd4, d4_method = False, None
        if xc == "r2scan-3c":
            xc = "r2scan"
            gcp_method = "r2scan-3c"
            d4_method = "r2scan-3c"

        if xc == "b97-3c":
            xc = "GGA_XC_B97_3C"
            gcp_method = "b97-3c"
            d3_method = "b97-3c"

        return cls(
            xc=xc,
            d3_method=d3_method,
            d4_method=d4_method,
            gcp_method=gcp_method,
            with_dftd3=with_dftd3,
            with_dftd4=with_dftd4,
            with_gcp=with_gcp,
        )
