"""Central configuration: paths, geography, product taxonomy and business assumptions.

A single source of truth shared by the simulator, the training pipeline, the
API and the test-suite. Every path can be overridden through an environment
variable (12-factor style), which is how the Docker image and the tests relocate
artefacts without touching code.

Business numbers in this module (freight costs, speeds, emission factors) are
explicit, documented *assumptions* - change them here and every report, plan and
API response follows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    return Path(os.getenv(var, str(default))).expanduser().resolve()


def _env_float(var: str, default: float) -> float:
    raw = os.getenv(var)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:  # fail fast on misconfiguration
        raise ValueError(f"Environment variable {var}={raw!r} is not a number") from exc


# --------------------------------------------------------------------------- #
# Product taxonomy                                                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CategorySpec:
    """Behavioural profile of a B2B product category.

    Used by the simulator to shape demand and by the dispatch planner to turn
    forecast order counts into freight weight.
    """

    code: str
    label: str
    kg_per_order: float  # average consignment weight of one order
    monsoon_multiplier: float  # demand multiplier during the south-west monsoon
    festive_uplift: float  # peak uplift in the three weeks before Diwali
    fy_end_uplift: float  # uplift in the last fortnight of the fiscal year (March)
    peak_doy: int  # day-of-year of the smooth annual peak


CATEGORIES: dict[str, CategorySpec] = {
    spec.code: spec
    for spec in (
        CategorySpec("IND_MACH", "Industrial Machinery & Tools", 140.0, 0.92, 0.10, 1.30, 75),
        CategorySpec("BLDG_MAT", "Building & Construction Material", 260.0, 0.62, 0.05, 1.20, 60),
        CategorySpec("TEXTILE", "Textiles, Yarn & Fabrics", 35.0, 1.00, 0.55, 1.05, 290),
        CategorySpec("ELEC", "Electrical Equipment & Supplies", 22.0, 0.95, 0.35, 1.15, 280),
        CategorySpec("PACKAGING", "Packaging Material", 48.0, 1.00, 0.60, 1.10, 285),
        CategorySpec("AGRI", "Agricultural Inputs & Equipment", 55.0, 1.35, 0.05, 1.00, 190),
    )
}


# --------------------------------------------------------------------------- #
# Geography: city hubs and hyper-local industrial zones                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ZoneSpec:
    zone_id: str
    city: str
    name: str
    lat: float
    lon: float


@dataclass(frozen=True)
class CitySpec:
    code: str
    name: str
    hub_lat: float
    hub_lon: float
    demand_scale: float  # relative market size
    category_affinity: tuple[tuple[str, float], ...]  # industrial-cluster specialisation
    zones: tuple[ZoneSpec, ...]

    def affinity(self, category: str) -> float:
        return dict(self.category_affinity).get(category, 1.0)


def _zones(city: str, rows: list[tuple[str, float, float]]) -> tuple[ZoneSpec, ...]:
    return tuple(
        ZoneSpec(f"{city}-Z{i:02d}", city, name, lat, lon) for i, (name, lat, lon) in enumerate(rows, start=1)
    )


# Coordinates are approximate centroids of well-known industrial areas,
# sufficient for city-scale routing; production would use geocoded addresses.
CITIES: dict[str, CitySpec] = {
    c.code: c
    for c in (
        CitySpec(
            "DEL",
            "Delhi NCR",
            28.652,
            77.141,
            1.40,
            (("ELEC", 1.8), ("PACKAGING", 1.3), ("TEXTILE", 1.2)),
            _zones(
                "DEL",
                [
                    ("Okhla Industrial Area", 28.530, 77.271),
                    ("Naraina Industrial Area", 28.630, 77.140),
                    ("Bawana Industrial Area", 28.795, 77.040),
                    ("Mayapuri Industrial Area", 28.637, 77.122),
                    ("Wazirpur Industrial Area", 28.699, 77.165),
                ],
            ),
        ),
        CitySpec(
            "MUM",
            "Mumbai MMR",
            19.296,
            73.063,
            1.40,
            (("PACKAGING", 1.6), ("ELEC", 1.3), ("IND_MACH", 1.2)),
            _zones(
                "MUM",
                [
                    ("Andheri MIDC", 19.118, 72.870),
                    ("Thane Wagle Estate", 19.200, 72.960),
                    ("TTC MIDC Turbhe", 19.075, 73.010),
                    ("Lower Parel", 18.995, 72.830),
                    ("Taloja MIDC", 19.065, 73.120),
                ],
            ),
        ),
        CitySpec(
            "AMD",
            "Ahmedabad",
            22.920,
            72.590,
            1.10,
            (("TEXTILE", 1.5), ("PACKAGING", 1.3), ("IND_MACH", 1.3)),
            _zones(
                "AMD",
                [
                    ("Naroda GIDC", 23.075, 72.665),
                    ("Vatva GIDC", 22.960, 72.630),
                    ("Odhav", 23.025, 72.660),
                    ("Sanand", 22.990, 72.380),
                    ("Changodar", 22.930, 72.440),
                ],
            ),
        ),
        CitySpec(
            "LDH",
            "Ludhiana",
            30.890,
            75.880,
            0.80,
            (("IND_MACH", 2.0), ("TEXTILE", 1.8)),
            _zones(
                "LDH",
                [
                    ("Focal Point", 30.875, 75.900),
                    ("Industrial Area A", 30.905, 75.860),
                    ("Gill Road", 30.880, 75.850),
                    ("Dhandari Kalan", 30.870, 75.930),
                    ("Sahnewal", 30.845, 75.985),
                ],
            ),
        ),
        CitySpec(
            "SRT",
            "Surat",
            21.180,
            72.960,
            0.90,
            (("TEXTILE", 2.6),),
            _zones(
                "SRT",
                [
                    ("Sachin GIDC", 21.090, 72.880),
                    ("Pandesara GIDC", 21.135, 72.840),
                    ("Udhna", 21.165, 72.845),
                    ("Varachha", 21.210, 72.870),
                    ("Katargam", 21.230, 72.830),
                ],
            ),
        ),
        CitySpec(
            "CBE",
            "Coimbatore",
            11.000,
            76.970,
            0.80,
            (("IND_MACH", 2.0), ("TEXTILE", 1.4)),
            _zones(
                "CBE",
                [
                    ("SIDCO Kurichi", 10.960, 76.960),
                    ("Peelamedu", 11.025, 77.000),
                    ("Ganapathy", 11.040, 76.970),
                    ("Malumichampatti", 10.905, 76.970),
                    ("Thudiyalur", 11.080, 76.940),
                ],
            ),
        ),
        CitySpec(
            "KOL",
            "Kolkata",
            22.560,
            88.350,
            1.00,
            (("AGRI", 1.5), ("BLDG_MAT", 1.3)),
            _zones(
                "KOL",
                [
                    ("Howrah", 22.590, 88.310),
                    ("Salt Lake Sector V", 22.575, 88.430),
                    ("Taratala", 22.505, 88.320),
                    ("Dankuni", 22.680, 88.290),
                    ("Kasba", 22.515, 88.385),
                ],
            ),
        ),
        CitySpec(
            "PUN",
            "Pune",
            18.620,
            73.830,
            1.00,
            (("IND_MACH", 1.8), ("ELEC", 1.2)),
            _zones(
                "PUN",
                [
                    ("Pimpri-Chinchwad", 18.630, 73.800),
                    ("Chakan", 18.760, 73.860),
                    ("Bhosari", 18.630, 73.850),
                    ("Hadapsar", 18.500, 73.930),
                    ("Talawade", 18.690, 73.770),
                ],
            ),
        ),
    )
}

ALL_ZONES: dict[str, ZoneSpec] = {z.zone_id: z for c in CITIES.values() for z in c.zones}


# --------------------------------------------------------------------------- #
# Fleet and routing assumptions (Indian urban B2B freight)                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FleetDefaults:
    logistics_attach_rate: float = 0.20  # share of orders shipping via the platform's partners
    capacity_kg: float = 2500.0  # light commercial vehicle (~2.5 t payload class)
    cost_per_km_inr: float = 26.0  # fuel + tyres + maintenance + toll share
    fixed_cost_inr: float = 1500.0  # driver day wage + daily share of vehicle EMI
    co2_kg_per_km: float = 0.38  # diesel LCV, ~7 km/l at 2.68 kg CO2 per litre
    avg_speed_kmph: float = 20.0  # congested city average for freight
    circuity_factor: float = 1.4  # road distance / straight-line distance
    service_min_per_drop: float = 20.0  # unloading + paperwork (e-way bill, GRN)
    shift_start_min: int = 8 * 60  # vehicles leave the hub from 08:00
    shift_end_min: int = 21 * 60  # and must be back by 21:00
    drop_tw_start_min: int = 9 * 60  # B2B receiving window opens 09:00
    drop_tw_end_min: int = 19 * 60  # and closes 19:00
    morning_window_end_min: int = 13 * 60  # some buyers receive only before 13:00


FLEET = FleetDefaults()


# --------------------------------------------------------------------------- #
# Runtime settings                                                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    """Filesystem layout and runtime knobs. Construct with :func:`get_settings`."""

    raw_dir: Path
    processed_dir: Path
    model_dir: Path
    reports_dir: Path
    routing_time_limit_s: float = 2.0
    random_seed: int = 42

    @property
    def raw_demand_path(self) -> Path:
        return self.raw_dir / "b2b_demand.csv.gz"

    @property
    def zones_path(self) -> Path:
        return self.raw_dir / "zones.csv"

    @property
    def history_path(self) -> Path:
        return self.processed_dir / "history.csv.gz"

    @property
    def data_quality_report_path(self) -> Path:
        return self.processed_dir / "data_quality_report.json"

    def ensure_dirs(self) -> None:
        for path in (self.raw_dir, self.processed_dir, self.model_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    """Build settings from environment variables with repository defaults."""
    return Settings(
        raw_dir=_env_path("DM_RAW_DIR", PROJECT_ROOT / "data" / "raw"),
        processed_dir=_env_path("DM_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"),
        model_dir=_env_path("DM_MODEL_DIR", PROJECT_ROOT / "models"),
        reports_dir=_env_path("DM_REPORTS_DIR", PROJECT_ROOT / "reports"),
        routing_time_limit_s=_env_float("DM_ROUTING_TIME_LIMIT_S", 2.0),
    )
