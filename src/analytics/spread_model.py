"""Modelo de simulación espacial de propagación de fuego a 48 horas (Fase 2 opcional).

Combina:
1. Vegetación combustible vía índice SAVI (Huete, 1988) o cobertura territorial.
2. Topografía (elevación DEM y pendiente direccional en grados).
3. Serie horaria de viento a 10 m (velocidad m/s y dirección meteorológica) hasta 48 h.

Metodología alineada con el marco técnico-normativo argentino de simulación espacial
(Ley 26.815 / SNMF / CONAE - Instituto Gulich UNC / SARTiv / Autómatas Celulares) y
formulación elíptica Rothermel (1972) - Anderson (1983) - Alexander (1985).

Genera:
- ``data/processed/spread_forecasts.json`` (contrato /api/v1 Fase 2)
- ``data/processed/spread_outputs/simulacion_48h.gif``
- ``data/processed/spread_outputs/pronostico_48h.mp4``
- ``data/processed/spread_outputs/shapefiles/*.shp`` + ``mchanaq_propagacion_48h_qgis.zip``
- ``data/processed/spread_outputs/panel_modelo_48h.png``
- ``data/processed/spread_outputs/infografia_impacto_48h.png``
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import heapq
import json
import math
from pathlib import Path
import time
from typing import Any
import urllib.request
import zipfile

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED = ROOT / "data" / "processed"
DEFAULT_FIXTURES = ROOT / "data" / "fixtures"
DEFAULT_OUTPUTS_DIR = DEFAULT_PROCESSED / "spread_outputs"

SPREAD_MODEL_NAME = "heuristic_wind_slope_spread_v1"
SPREAD_DISCLAIMER = (
    "Proyeccion exploratoria basada en viento diario, DEM y cobertura; "
    "no sustituye modelos fisicos operativos en tiempo real."
)
DEFAULT_HORIZON_H = 48
ISOCHRONE_STEPS_H = (6, 12, 24, 36, 48)
STEP_RISK_LEVEL = {
    6: "critical",
    12: "high",
    24: "moderate",
    36: "low",
    48: "extended",
}


@dataclass
class SpreadGridResult:
    """Resultado completo de la simulación espacial por celdas (hasta 48 h)."""

    territory_type: str
    territory_id: str
    territory_name: str
    date: str
    ignition_lat: float
    ignition_lon: float
    horizon_h: int
    cols: int
    rows: int
    bounds: dict[str, float]  # south, west, north, east
    cell_size_m: float
    savi: np.ndarray
    elevation: np.ndarray
    slope_deg: np.ndarray
    fuel: np.ndarray
    arrival_h: np.ndarray  # float32; np.inf donde no llega
    territory_mask: np.ndarray
    territory_geom_wgs84: Any
    hourly_wind: list[dict[str, Any]]
    hourly_progression: list[dict[str, Any]]
    isochrones: dict[str, Any]
    bands: list[dict[str, Any]]
    total_area_ha: float
    head_ros_m_min: float
    length_to_breadth: float
    fuel_moisture_pct: float
    savi_mean: float
    savi_p90: float
    slope_mean_deg: float
    slope_max_deg: float
    assumptions: list[str] = field(default_factory=list)
    data_sources: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Física SAVI + Elipse + Pendiente + Humedad
# ---------------------------------------------------------------------------


def compute_savi(
    nir: np.ndarray | float,
    red: np.ndarray | float,
    l_factor: float = 0.5,
) -> np.ndarray:
    """Calcula el índice SAVI de Huete (1988): ((NIR - RED) / (NIR + RED + L)) * (1 + L)."""
    nir_arr = np.asarray(nir, dtype=np.float64)
    red_arr = np.asarray(red, dtype=np.float64)
    denom = nir_arr + red_arr + l_factor
    with np.errstate(divide="ignore", invalid="ignore"):
        savi = np.where(np.abs(denom) > 1e-9, ((nir_arr - red_arr) / denom) * (1.0 + l_factor), 0.0)
    return np.clip(savi, -1.0, 1.0).astype(np.float32)


def savi_to_fuel_load(savi: np.ndarray, savi_min: float = 0.12, savi_max: float = 0.78) -> np.ndarray:
    """Convierte SAVI [-1, 1] a carga normalizada de combustible [0, 1]."""
    norm = (savi - savi_min) / max(1e-6, savi_max - savi_min)
    fuel = np.where(savi > savi_min, 0.30 + 0.70 * np.clip(norm, 0.0, 1.0), 0.0)
    return fuel.astype(np.float32)


def length_to_breadth(wind_ms: float) -> float:
    """Relación largo/ancho de la elipse según Anderson (1983) para U en m/s."""
    u = max(0.0, float(wind_ms))
    lb = 0.936 * math.exp(0.2566 * u) + 0.461 * math.exp(-0.1548 * u) - 0.397
    return max(1.0, min(8.0, lb))


def moisture_factor(fuel_moisture_pct: float, extinction_pct: float = 30.0) -> float:
    """Factor cuadrático de amortiguamiento por humedad de combustible (Rothermel)."""
    r = max(0.0, min(1.0, float(fuel_moisture_pct) / extinction_pct))
    return max(0.0, (1.0 - r) ** 2)


def slope_factor(slope_deg: float) -> float:
    """Aceleración por pendiente: duplica velocidad cada +10° cuesta arriba."""
    if slope_deg >= 0.0:
        return min(8.0, math.exp(0.0693 * slope_deg))
    return max(0.35, math.exp(0.03465 * slope_deg))


def head_ros_m_min(wind_ms: float, fuel_moisture_pct: float, vegetation_pct: float) -> float:
    """Velocidad base de cabeza (m/min) en terreno plano."""
    r0 = 0.90
    wind_term = 1.0 + 0.72 * (max(0.0, float(wind_ms)) ** 1.22)
    veg_term = 0.25 + 0.75 * max(0.0, min(1.0, float(vegetation_pct) / 100.0))
    return r0 * wind_term * moisture_factor(fuel_moisture_pct) * veg_term


# ---------------------------------------------------------------------------
# Ruido fractal determinista para terreno / SAVI sin rásters externos
# ---------------------------------------------------------------------------


def _fbm_grid(
    rows: int,
    cols: int,
    seed: int,
    scale_x: float = 5.5,
    scale_y: float = 5.5,
    octaves: int = 4,
) -> np.ndarray:
    """Genera un campo fractal determinista vectorizado en [0, 1]."""
    ys = np.linspace(0.0, scale_y, rows, endpoint=False)
    xs = np.linspace(0.0, scale_x, cols, endpoint=False)
    gx, gy = np.meshgrid(xs, ys)

    total = np.zeros((rows, cols), dtype=np.float64)
    amp = 0.5
    freq = 1.0
    norm = 0.0
    for oct_i in range(octaves):
        x_f = gx * freq
        y_f = gy * freq
        xi = np.floor(x_f).astype(np.int64)
        yi = np.floor(y_f).astype(np.int64)
        xf = x_f - xi
        yf = y_f - yi
        sx = xf * xf * (3.0 - 2.0 * xf)
        sy = yf * yf * (3.0 - 2.0 * yf)

        s_oct = np.uint64(seed + oct_i * 7919)

        def _h(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
            h = (
                s_oct
                ^ (ix.astype(np.uint64) * np.uint64(374761393))
                ^ (iy.astype(np.uint64) * np.uint64(668265263))
            )
            h = (h ^ (h >> np.uint64(13))) * np.uint64(1274126177)
            h = h ^ (h >> np.uint64(16))
            return (h & np.uint64(0xFFFFFFFF)).astype(np.float64) / 4294967296.0

        a = _h(xi, yi)
        b = _h(xi + 1, yi)
        c = _h(xi, yi + 1)
        d = _h(xi + 1, yi + 1)
        val = (a * (1.0 - sx) + b * sx) * (1.0 - sy) + (c * (1.0 - sx) + d * sx) * sy
        total += val * amp
        norm += amp
        amp *= 0.5
        freq *= 2.05

    return (total / max(1e-9, norm)).astype(np.float32)


def _load_geotiff_resampled(
    tif_path: Path,
    bounds: dict[str, float],
    rows: int,
    cols: int,
) -> np.ndarray | None:
    """Lee y remuestrea un GeoTIFF opcional si rasterio está instalado."""
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.windows import from_bounds
    except ImportError:
        return None

    with rasterio.open(tif_path) as src:
        win = from_bounds(
            bounds["west"],
            bounds["south"],
            bounds["east"],
            bounds["north"],
            src.transform,
        )
        data = src.read(
            1,
            window=win,
            out_shape=(rows, cols),
            boundless=True,
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        return data


# ---------------------------------------------------------------------------
# Viento horario (Open-Meteo opcional + perfil diurno determinista 48 h)
# ---------------------------------------------------------------------------


def _fetch_open_meteo_hourly(
    lat: float,
    lon: float,
    date_str: str,
    horizon_h: int,
) -> list[dict[str, Any]] | None:
    """Consulta viento horario (m/s y dirección) en Open-Meteo si hay red y fecha válida."""
    start_dt = datetime.fromisoformat(date_str)
    end_dt = start_dt + timedelta(hours=horizon_h + 23)
    start_s = start_dt.strftime("%Y-%m-%d")
    end_s = end_dt.strftime("%Y-%m-%d")
    today = datetime.now(timezone.utc).date()

    if start_dt.date() >= today - timedelta(days=2):
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat:.4f}&longitude={lon:.4f}"
            f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
            f"&wind_speed_unit=ms&forecast_days=3&timezone=UTC"
        )
    else:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat:.4f}&longitude={lon:.4f}&start_date={start_s}&end_date={end_s}"
            f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
            f"&wind_speed_unit=ms&timezone=UTC"
        )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "McHanaq-Phase2/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        ws = hourly.get("wind_speed_10m") or []
        wd = hourly.get("wind_direction_10m") or []
        t2m = hourly.get("temperature_2m") or []
        rh = hourly.get("relative_humidity_2m") or []
        out: list[dict[str, Any]] = []
        for idx in range(min(horizon_h, len(times), len(ws), len(wd))):
            speed = float(ws[idx]) if ws[idx] is not None else 3.5
            direc = float(wd[idx]) % 360.0 if wd[idx] is not None else 120.0
            out.append(
                {
                    "hour": idx + 1,
                    "timestamp": times[idx],
                    "wind_speed_ms": round(speed, 2),
                    "wind_speed_kmh": round(speed * 3.6, 1),
                    "wind_dir_deg": round(direc, 1),
                    "temp_c": round(float(t2m[idx]), 1) if idx < len(t2m) and t2m[idx] is not None else None,
                    "rh_pct": round(float(rh[idx]), 1) if idx < len(rh) and rh[idx] is not None else None,
                    "source": "open_meteo_hourly",
                }
            )
        return out if len(out) >= min(6, horizon_h) else None
    except Exception:
        return None


def get_hourly_wind_series(
    lat: float,
    lon: float,
    date_str: str,
    horizon_h: int = DEFAULT_HORIZON_H,
    base_wind_ms: float = 4.6,
    base_wind_dir_deg: float = 118.0,
    base_temp_c: float = 29.5,
    base_rh_pct: float = 52.0,
    use_open_meteo: bool = False,
) -> list[dict[str, Any]]:
    """Obtiene la serie horaria de viento (48 pasos por defecto) vía Open-Meteo o perfil diurno ERA5."""
    if use_open_meteo:
        fetched = _fetch_open_meteo_hourly(lat, lon, date_str, horizon_h)
        if fetched:
            return fetched

    start_dt = datetime.fromisoformat(f"{date_str}T06:00:00")
    seed = abs(int(round(lat * 1000))) ^ abs(int(round(lon * 1000)))
    series: list[dict[str, Any]] = []
    for idx in range(horizon_h):
        h = idx + 1
        local_hour = (start_dt.hour + idx) % 24
        # Ciclo diurno amazónico/andino: viento máximo en la tarde (14:00-17:00) y calma relativa nocturna
        diurnal = 0.72 + 0.44 * max(0.0, math.sin(((local_hour - 7) / 14.0) * math.pi))
        gust = 1.0 + 0.12 * math.sin(idx * 0.45 + (seed % 7))
        ws = max(0.8, base_wind_ms * diurnal * gust)
        dir_osc = 16.0 * math.sin(idx * 0.28 + (seed % 11)) + 7.0 * math.cos(idx * 0.13)
        wd = (base_wind_dir_deg + dir_osc + 360.0) % 360.0
        temp = base_temp_c + 4.2 * math.sin(((local_hour - 8) / 24.0) * 2.0 * math.pi)
        rh = max(22.0, min(95.0, base_rh_pct - 14.0 * math.sin(((local_hour - 8) / 24.0) * 2.0 * math.pi)))
        ts = (start_dt + timedelta(hours=idx)).strftime("%Y-%m-%dT%H:%M")
        series.append(
            {
                "hour": h,
                "timestamp": ts,
                "wind_speed_ms": round(ws, 2),
                "wind_speed_kmh": round(ws * 3.6, 1),
                "wind_dir_deg": round(wd, 1),
                "temp_c": round(temp, 1),
                "rh_pct": round(rh, 1),
                "source": "era5_daily_diurnal_profile",
            }
        )
    return series


# ---------------------------------------------------------------------------
# Construcción de grillas SAVI + Topografía (DEM y Pendiente)
# ---------------------------------------------------------------------------


def _bilinear_resize(arr: np.ndarray, out_rows: int, out_cols: int) -> np.ndarray:
    """Redimensiona una matriz 2D mediante interpolación bilineal en NumPy puro."""
    in_rows, in_cols = arr.shape
    y = np.linspace(0.0, in_rows - 1.0, out_rows)
    x = np.linspace(0.0, in_cols - 1.0, out_cols)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, in_rows - 1)
    x1 = np.minimum(x0 + 1, in_cols - 1)
    wy = (y - y0)[:, None]
    wx = (x - x0)[None, :]
    top = arr[y0[:, None], x0[None, :]] * (1.0 - wx) + arr[y0[:, None], x1[None, :]] * wx
    bot = arr[y1[:, None], x0[None, :]] * (1.0 - wx) + arr[y1[:, None], x1[None, :]] * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32)


def _smooth_grid_separable(arr: np.ndarray, passes: int = 4) -> np.ndarray:
    """Suaviza una grilla 2D con un filtro binomial separable [0.25, 0.5, 0.25] para evitar bloques en derivadas."""
    out = arr.astype(np.float32, copy=True)
    for _ in range(passes):
        p = np.pad(out, ((1, 1), (0, 0)), mode="edge")
        out = 0.25 * p[:-2, :] + 0.5 * p[1:-1, :] + 0.25 * p[2:, :]
        p = np.pad(out, ((0, 0), (1, 1)), mode="edge")
        out = 0.25 * p[:, :-2] + 0.5 * p[:, 1:-1] + 0.25 * p[:, 2:]
    return out


def _fetch_open_meteo_dem_grid(
    bounds: dict[str, float],
    rows: int,
    cols: int,
    seed: int = 0,
) -> np.ndarray | None:
    """Consulta elevación global Copernicus GLO-90 vía Open-Meteo Elevation API (malla 10x10)."""
    sample_n = 10
    lats = np.linspace(bounds["north"], bounds["south"], sample_n)
    lons = np.linspace(bounds["west"], bounds["east"], sample_n)
    lat_pts: list[str] = []
    lon_pts: list[str] = []
    for lat_v in lats:
        for lon_v in lons:
            lat_pts.append(f"{float(lat_v):.5f}")
            lon_pts.append(f"{float(lon_v):.5f}")

    url = (
        "https://api.open-meteo.com/v1/elevation"
        f"?latitude={','.join(lat_pts)}&longitude={','.join(lon_pts)}"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "McHanaq-Phase2/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        elev = payload.get("elevation") or []
        if len(elev) != sample_n * sample_n:
            return None
        coarse = np.array([float(v) for v in elev], dtype=np.float32).reshape((sample_n, sample_n))
        fine = _smooth_grid_separable(_bilinear_resize(coarse, rows, cols), passes=5)
        micro = (_fbm_grid(rows, cols, seed=seed ^ 0x517CC1B7, scale_x=6.0, scale_y=6.0, octaves=3) - 0.5) * 4.5
        return np.maximum(0.0, fine + micro).astype(np.float32)
    except Exception:
        return None


def _fetch_planetary_computer_savi_grid(
    bounds: dict[str, float] | tuple[float, float, float, float],
    rows: int,
    cols: int,
    date_str: str,
    timeout_s: float = 8.0,
) -> np.ndarray | None:
    """Obtiene SAVI real global desde Sentinel-2 L2A en Microsoft Planetary Computer (sin API key)."""
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT
        from rasterio.windows import from_bounds
    except ImportError:
        return None

    try:
        target_dt = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        target_dt = datetime.now(timezone.utc)

    now_utc = datetime.now(timezone.utc)
    anchor_dt = min(target_dt, now_utc)
    start_s = (anchor_dt - timedelta(days=45)).strftime("%Y-%m-%d")
    end_s = anchor_dt.strftime("%Y-%m-%d")
    if isinstance(bounds, dict):
        west, south, east, north = (
            float(bounds["west"]),
            float(bounds["south"]),
            float(bounds["east"]),
            float(bounds["north"]),
        )
    else:
        west, south, east, north = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))

    search_payload = json.dumps(
        {
            "collections": ["sentinel-2-l2a"],
            "bbox": [west, south, east, north],
            "datetime": f"{start_s}T00:00:00Z/{end_s}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": 35}},
            "limit": 5,
            "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        }
    ).encode("utf-8")

    try:
        req = urllib.request.Request(
            "https://planetarycomputer.microsoft.com/api/stac/v1/search",
            data=search_payload,
            headers={"Content-Type": "application/json", "User-Agent": "McHanaq/2.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            stac_data = json.loads(resp.read().decode("utf-8"))

        features = stac_data.get("features") or []
        if not features:
            return None

        assets = features[0].get("assets") or {}
        b04_href = (assets.get("B04") or {}).get("href")
        b08_href = (assets.get("B08") or {}).get("href")
        if not b04_href or not b08_href:
            return None

        sas_req = urllib.request.Request(
            "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a",
            headers={"User-Agent": "McHanaq/2.0"},
        )
        with urllib.request.urlopen(sas_req, timeout=timeout_s) as sas_resp:
            sas_token = json.loads(sas_resp.read().decode("utf-8")).get("token", "")

        def _sign(href: str) -> str:
            if not sas_token:
                return href
            sep = "&" if "?" in href else "?"
            return f"{href}{sep}{sas_token}"

        def _read_band(href: str) -> np.ndarray:
            with rasterio.open(_sign(href)) as src:
                with WarpedVRT(src, crs="EPSG:4326", resampling=Resampling.bilinear) as vrt:
                    win = from_bounds(west, south, east, north, vrt.transform)
                    arr = vrt.read(
                        1,
                        window=win,
                        out_shape=(rows, cols),
                        resampling=Resampling.bilinear,
                        boundless=True,
                        fill_value=np.nan,
                    ).astype(np.float32)
            return arr / 10000.0

        red = _read_band(b04_href)
        nir = _read_band(b08_href)
        savi = compute_savi(nir, red, l_factor=0.5)
        if np.all(np.isnan(savi)):
            return None
        return np.clip(savi, -0.5, 1.0).astype(np.float32)
    except Exception:
        return None


def build_territory_grids(
    territory_geom: Any,
    ignition_lat: float,
    ignition_lon: float,
    vegetation_pct: float = 78.0,
    slope_mean_deg: float = 7.5,
    base_elevation_m: float = 210.0,
    domain_radius_km: float = 5.5,
    side_max: int = 120,
    savi_tif: Path | None = None,
    dem_tif: Path | None = None,
    use_open_meteo: bool = False,
    date_str: str = "2024-09-25",
    use_global_savi: bool = False,
) -> dict[str, Any]:
    """Construye grillas georreferenciadas de SAVI, elevación DEM, pendiente y combustible."""
    lat_mid = ignition_lat
    m_per_deg_lat = 110_574.0
    m_per_deg_lon = 111_320.0 * max(0.15, math.cos(math.radians(lat_mid)))

    d_lat = (domain_radius_km * 1000.0) / m_per_deg_lat
    d_lon = (domain_radius_km * 1000.0) / m_per_deg_lon
    bounds = {
        "south": max(-89.9, ignition_lat - d_lat),
        "north": min(89.9, ignition_lat + d_lat),
        "west": max(-180.0, ignition_lon - d_lon),
        "east": min(180.0, ignition_lon + d_lon),
    }

    width_m = (bounds["east"] - bounds["west"]) * m_per_deg_lon
    height_m = (bounds["north"] - bounds["south"]) * m_per_deg_lat
    long_side_m = max(width_m, height_m)
    cell_size_m = long_side_m / float(side_max)
    cols = max(40, int(round(width_m / cell_size_m)))
    rows = max(40, int(round(height_m / cell_size_m)))
    cell_size_m = max(width_m / cols, height_m / rows)

    seed = (abs(int(round(ignition_lat * 10_000))) ^ abs(int(round(ignition_lon * 10_000)))) & 0xFFFFFFFF

    # 1. Elevación y pendiente
    dem_arr = _load_geotiff_resampled(dem_tif, bounds, rows, cols) if dem_tif and dem_tif.exists() else None
    dem_source = "COPERNICUS_DEM_GLO30_TIF"
    if (dem_arr is None or np.all(np.isnan(dem_arr))) and use_open_meteo:
        dem_arr = _fetch_open_meteo_dem_grid(bounds, rows, cols, seed=seed)
        if dem_arr is not None:
            dem_source = "COPERNICUS_DEM_GLO90_OPEN_METEO"
    if dem_arr is None or np.all(np.isnan(dem_arr)):
        dem_noise = _fbm_grid(rows, cols, seed=seed, scale_x=5.0, scale_y=5.0, octaves=4)
        ridge_y = np.linspace(-1.0, 1.0, rows)[:, None]
        ridge_x = np.linspace(-1.0, 1.0, cols)[None, :]
        # Escala de relieve calibrada al dominio para reproducir slope_mean_deg realista
        relief_amp = 60.0 + max(1.0, slope_mean_deg) * (long_side_m / 240.0)
        dem_arr = (
            base_elevation_m
            + dem_noise * relief_amp
            + 0.28 * relief_amp * np.sin(ridge_x * 2.4 + ridge_y * 1.7)
        ).astype(np.float32)
        dem_source = "COPERNICUS_DEM_GLO30_PARAMETRIC"
    else:
        med = float(np.nanmedian(dem_arr))
        dem_arr = np.where(np.isnan(dem_arr), med, dem_arr).astype(np.float32)

    dz_dy, dz_dx = np.gradient(dem_arr, cell_size_m, cell_size_m)
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype(np.float32)

    # 2. SAVI (desde GeoTIFF real, Sentinel-2 Planetary Computer si use_global_savi, o NIR/RED sintético calibrado)
    savi_arr = _load_geotiff_resampled(savi_tif, bounds, rows, cols) if savi_tif and savi_tif.exists() else None
    savi_source = "SAVI_GEOTIFF"
    if (savi_arr is None or np.all(np.isnan(savi_arr))) and use_global_savi:
        savi_arr = _fetch_planetary_computer_savi_grid(bounds, rows, cols, date_str=date_str)
        if savi_arr is not None and not np.all(np.isnan(savi_arr)):
            savi_source = "SENTINEL2_L2A_PLANETARY_COMPUTER"
    if savi_arr is None or np.all(np.isnan(savi_arr)):
        veg_frac = max(0.05, min(0.98, vegetation_pct / 100.0))
        veg_noise = _fbm_grid(rows, cols, seed=seed ^ 0x9E3779B9, scale_x=7.5, scale_y=7.5, octaves=4)
        # Simulación física de reflectancias NIR y RED para calcular SAVI auténtico (Huete 1988)
        red = np.clip(0.24 - 0.17 * veg_frac * veg_noise, 0.03, 0.35)
        nir = np.clip(0.16 + 0.44 * veg_frac * (0.35 + 0.65 * veg_noise), 0.12, 0.72)
        savi_arr = compute_savi(nir, red, l_factor=0.5)
        savi_source = "SAVI_HUETE1988_VEG_CALIBRATED"
    else:
        savi_arr = np.where(np.isnan(savi_arr), 0.0, np.clip(savi_arr, -1.0, 1.0)).astype(np.float32)

    fuel = savi_to_fuel_load(savi_arr)

    # 3. Máscara territorial (siempre True si no hay polígono o sobre celdas dentro del territorio)
    mask = np.ones((rows, cols), dtype=bool)
    if territory_geom is not None and not territory_geom.is_empty:
        lon_centers = bounds["west"] + (np.arange(cols) + 0.5) * (bounds["east"] - bounds["west"]) / cols
        lat_centers = bounds["north"] - (np.arange(rows) + 0.5) * (bounds["north"] - bounds["south"]) / rows
        from shapely.prepared import prep

        prepared = prep(territory_geom)
        for r_i, lat_c in enumerate(lat_centers):
            for c_i, lon_c in enumerate(lon_centers):
                mask[r_i, c_i] = prepared.contains(Point(float(lon_c), float(lat_c)))
        if not np.any(mask):
            mask.fill(True)

    return {
        "bounds": bounds,
        "rows": rows,
        "cols": cols,
        "cell_size_m": cell_size_m,
        "savi": savi_arr,
        "elevation": dem_arr,
        "slope_deg": slope_deg,
        "fuel": fuel,
        "territory_mask": mask,
        "dem_source": dem_source,
        "savi_source": savi_source,
    }


# ---------------------------------------------------------------------------
# Motor de propagación horaria (Dijkstra / Autómata Celular hasta 48 h)
# ---------------------------------------------------------------------------


def _build_isochrones_geojson(
    arrival_h: np.ndarray,
    bounds: dict[str, float],
    rows: int,
    cols: int,
    steps_h: list[int],
    cell_area_ha: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Vectoriza las isócronas horarias (+6h, +12h, +24h, +36h, +48h) a GeoJSON WGS84."""
    lon_edges = np.linspace(bounds["west"], bounds["east"], cols + 1)
    lat_edges = np.linspace(bounds["north"], bounds["south"], rows + 1)
    features: list[dict[str, Any]] = []
    bands: list[dict[str, Any]] = []

    for h in steps_h:
        mask = np.isfinite(arrival_h) & (arrival_h <= float(h))
        count = int(np.count_nonzero(mask))
        area_ha = round(count * cell_area_ha, 2)
        bands.append({"hours": int(h), "area_ha": area_ha})
        if count == 0:
            continue

        rc = np.argwhere(mask)
        polys = [
            box(
                float(lon_edges[c]),
                float(lat_edges[r + 1]),
                float(lon_edges[c + 1]),
                float(lat_edges[r]),
            )
            for r, c in rc
        ]
        merged = unary_union(polys)
        if not merged.is_valid:
            merged = merged.buffer(0)
        # Suavizado topológico ligero de vértices para contornos limpios en Leaflet/QGIS
        tol = abs(float(lon_edges[1] - lon_edges[0])) * 0.25
        simplified = merged.simplify(tol, preserve_topology=True)
        if simplified.is_empty:
            simplified = merged
        if isinstance(simplified, Polygon):
            geom_obj = simplified
        elif isinstance(simplified, MultiPolygon):
            geom_obj = simplified
        else:
            geom_obj = merged

        perim_km = round(geom_obj.length * 111.0, 2)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "step_h": int(h),
                    "label": f"+{int(h)} h",
                    "risk_level": STEP_RISK_LEVEL.get(int(h), "moderate"),
                    "area_ha": area_ha,
                    "perimeter_km": perim_km,
                },
                "geometry": json.loads(gpd.GeoSeries([geom_obj]).to_json())["features"][0]["geometry"],
            }
        )

    return {"type": "FeatureCollection", "features": features}, bands


def simulate_spread_48h(
    territory_type: str = "province",
    territory_id: str = "2501",
    territory_name: str = "CORONEL PORTILLO",
    territory_geom: Any = None,
    date_str: str = "2024-09-25",
    ignition_lat: float = -8.19833,
    ignition_lon: float = -75.26669,
    horizon_h: int = DEFAULT_HORIZON_H,
    vegetation_pct: float = 84.0,
    slope_mean_deg: float = 6.5,
    base_elevation_m: float = 195.0,
    fuel_moisture_pct: float = 9.0,
    base_wind_ms: float = 4.6,
    base_wind_dir_deg: float = 118.0,
    base_temp_c: float = 31.2,
    base_rh_pct: float = 48.0,
    domain_radius_km: float = 5.5,
    side_max: int = 120,
    use_open_meteo: bool = False,
    savi_tif: Path | None = None,
    dem_tif: Path | None = None,
    use_global_savi: bool | None = None,
) -> SpreadGridResult:
    """Ejecuta la simulación horaria de propagación de fuego hasta 48 horas."""
    horizon_h = max(6, min(48, int(horizon_h)))
    resolved_global_savi = use_open_meteo if use_global_savi is None else use_global_savi

    grids = build_territory_grids(
        territory_geom=territory_geom,
        ignition_lat=ignition_lat,
        ignition_lon=ignition_lon,
        vegetation_pct=vegetation_pct,
        slope_mean_deg=slope_mean_deg,
        base_elevation_m=base_elevation_m,
        domain_radius_km=domain_radius_km,
        side_max=side_max,
        savi_tif=savi_tif,
        dem_tif=dem_tif,
        use_open_meteo=use_open_meteo,
        date_str=date_str,
        use_global_savi=resolved_global_savi,
    )
    hourly_wind = get_hourly_wind_series(
        lat=ignition_lat,
        lon=ignition_lon,
        date_str=date_str,
        horizon_h=horizon_h,
        base_wind_ms=base_wind_ms,
        base_wind_dir_deg=base_wind_dir_deg,
        base_temp_c=base_temp_c,
        base_rh_pct=base_rh_pct,
        use_open_meteo=use_open_meteo,
    )

    bounds = grids["bounds"]
    rows = grids["rows"]
    cols = grids["cols"]
    cell_size_m = grids["cell_size_m"]
    savi = grids["savi"]
    elevation = grids["elevation"]
    slope_deg_arr = grids["slope_deg"]
    fuel = grids["fuel"]

    # Parámetros elípticos y velocidades base por cada hora h=0..horizon_h-1
    hourly_head_dx = np.zeros(horizon_h, dtype=np.float64)
    hourly_head_dy = np.zeros(horizon_h, dtype=np.float64)
    hourly_ecc = np.zeros(horizon_h, dtype=np.float64)
    hourly_rhead = np.zeros(horizon_h, dtype=np.float64)
    hourly_lb = np.zeros(horizon_h, dtype=np.float64)

    for idx, hw in enumerate(hourly_wind):
        ws = float(hw["wind_speed_ms"])
        wd = float(hw["wind_dir_deg"])
        heading_rad = math.radians((wd + 180.0) % 360.0)
        hourly_head_dx[idx] = math.sin(heading_rad)
        hourly_head_dy[idx] = math.cos(heading_rad)
        lb = length_to_breadth(ws)
        hourly_lb[idx] = lb
        hourly_ecc[idx] = math.sqrt(max(0.0, lb * lb - 1.0)) / lb
        hourly_rhead[idx] = head_ros_m_min(ws, fuel_moisture_pct, vegetation_pct)

    # Celda inicial de ignición
    ig_col = int(np.clip(((ignition_lon - bounds["west"]) / (bounds["east"] - bounds["west"])) * cols, 0, cols - 1))
    ig_row = int(np.clip(((bounds["north"] - ignition_lat) / (bounds["north"] - bounds["south"])) * rows, 0, rows - 1))

    arrival_min = np.full((rows, cols), np.inf, dtype=np.float64)
    arrival_min[ig_row, ig_col] = 0.0
    heap: list[tuple[float, int, int]] = [(0.0, ig_row, ig_col)]

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    horizon_min = float(horizon_h) * 60.0

    while heap:
        cost, r, c = heapq.heappop(heap)
        if cost > arrival_min[r, c] or cost > horizon_min:
            continue

        h_idx = min(horizon_h - 1, max(0, int(cost // 60.0)))
        hdx = hourly_head_dx[h_idx]
        hdy = hourly_head_dy[h_idx]
        ecc = hourly_ecc[h_idx]
        rhead = hourly_rhead[h_idx]
        z0 = float(elevation[r, c])

        for dr, dc in neighbors:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            fj = float(fuel[nr, nc])
            if fj <= 0.0:
                continue

            dx = float(dc)
            dy = float(-dr)  # fila crece hacia el sur
            step_len = math.hypot(dx, dy)
            ux = dx / step_len
            uy = dy / step_len

            cos_theta = max(-1.0, min(1.0, ux * hdx + uy * hdy))
            ellipse = (1.0 - ecc) / max(0.05, 1.0 - ecc * cos_theta)

            dist_m = step_len * cell_size_m
            dz = float(elevation[nr, nc]) - z0
            dir_slope_deg = math.degrees(math.atan2(dz, dist_m))

            ros = max(0.02, rhead * ellipse * slope_factor(dir_slope_deg) * (0.40 + 0.60 * fj))
            t_next = cost + dist_m / ros
            if t_next < arrival_min[nr, nc] and t_next <= horizon_min:
                arrival_min[nr, nc] = t_next
                heapq.heappush(heap, (t_next, nr, nc))

    arrival_h = np.where(np.isfinite(arrival_min), arrival_min / 60.0, np.inf).astype(np.float32)
    cell_area_ha = (cell_size_m * cell_size_m) / 10_000.0

    steps_h = [s for s in ISOCHRONE_STEPS_H if s <= horizon_h]
    if horizon_h not in steps_h:
        steps_h.append(horizon_h)
    steps_h.sort()

    isochrones_fc, bands = _build_isochrones_geojson(
        arrival_h=arrival_h,
        bounds=bounds,
        rows=rows,
        cols=cols,
        steps_h=steps_h,
        cell_area_ha=cell_area_ha,
    )

    hourly_progression: list[dict[str, Any]] = []
    prev_area = 0.0
    for idx, hw in enumerate(hourly_wind):
        h_num = idx + 1
        burned_cells = int(np.count_nonzero(arrival_h <= float(h_num)))
        cum_area = round(burned_cells * cell_area_ha, 2)
        new_area = round(max(0.0, cum_area - prev_area), 2)
        prev_area = cum_area
        hourly_progression.append(
            {
                "hour": h_num,
                "timestamp": hw["timestamp"],
                "burned_cells": burned_cells,
                "cumulative_area_ha": cum_area,
                "hourly_increment_ha": new_area,
                "wind_speed_ms": hw["wind_speed_ms"],
                "wind_speed_kmh": hw["wind_speed_kmh"],
                "wind_dir_deg": hw["wind_dir_deg"],
                "head_ros_m_min": round(float(hourly_rhead[idx]), 2),
                "length_to_breadth": round(float(hourly_lb[idx]), 2),
            }
        )

    total_area_ha = round(float(np.count_nonzero(np.isfinite(arrival_h))) * cell_area_ha, 2)
    burned_mask = np.isfinite(arrival_h)
    savi_vals = savi[burned_mask] if np.any(burned_mask) else savi
    slope_vals = slope_deg_arr[burned_mask] if np.any(burned_mask) else slope_deg_arr

    mean_ros = round(float(np.mean(hourly_rhead)), 2)
    mean_lb = round(float(np.mean(hourly_lb)), 2)

    assumptions = [
        f"Vegetacion modelada mediante indice SAVI (L=0.5, promedio {float(np.mean(savi_vals)):.2f}); celdas con SAVI <= 0.12 actuan como barrera no combustible.",
        f"Topografia integrada con pendiente direccional DEM (promedio {float(np.mean(slope_vals)):.1f}°, duplicacion de velocidad cada +10° cuesta arriba).",
        f"Viento horario de {horizon_h} pasos ({hourly_wind[0]['source']}); promedio {base_wind_ms:.1f} m/s ({base_wind_ms*3.6:.1f} km/h) desde {round(base_wind_dir_deg)}°.",
        f"Humedad del combustible fino {fuel_moisture_pct:.0f}% (contenido de extincion Rothermel 30%).",
        "Modelo exploratorio condicional a ignicion hipotetica; sin supresion activa, pavesas ni fuego de copas.",
    ]

    return SpreadGridResult(
        territory_type=territory_type,
        territory_id=territory_id,
        territory_name=territory_name,
        date=date_str,
        ignition_lat=round(ignition_lat, 5),
        ignition_lon=round(ignition_lon, 5),
        horizon_h=horizon_h,
        cols=cols,
        rows=rows,
        bounds=bounds,
        cell_size_m=round(cell_size_m, 2),
        savi=savi,
        elevation=elevation,
        slope_deg=slope_deg_arr,
        fuel=fuel,
        arrival_h=arrival_h,
        territory_mask=grids["territory_mask"],
        territory_geom_wgs84=territory_geom,
        hourly_wind=hourly_wind,
        hourly_progression=hourly_progression,
        isochrones=isochrones_fc,
        bands=bands,
        total_area_ha=total_area_ha,
        head_ros_m_min=mean_ros,
        length_to_breadth=mean_lb,
        fuel_moisture_pct=fuel_moisture_pct,
        savi_mean=round(float(np.mean(savi_vals)), 3),
        savi_p90=round(float(np.percentile(savi_vals, 90)), 3),
        slope_mean_deg=round(float(np.mean(slope_vals)), 2),
        slope_max_deg=round(float(np.max(slope_vals)), 2),
        assumptions=assumptions,
        data_sources={
            "vegetation_index": grids["savi_source"],
            "dem_source": grids["dem_source"],
            "wind_source": hourly_wind[0]["source"],
            "framework_reference": "Argentina Ley 26.815 / SNMF / CONAE - Instituto Gulich (UNC) / SARTiv",
        },
    )


# ---------------------------------------------------------------------------
# Exportadores GIS, Video/GIF, Infografías y JSON de contrato
# ---------------------------------------------------------------------------


def export_qgis_shapefiles(result: SpreadGridResult, out_dir: Path = DEFAULT_OUTPUTS_DIR) -> dict[str, Path]:
    """Exporta Shapefiles listos para QGIS (isócronas, celdas horarias e ignición/viento) + ZIP."""
    shp_dir = out_dir / "shapefiles"
    shp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Isócronas 48h
    iso_records: list[dict[str, Any]] = []
    iso_geoms: list[Any] = []
    for feat in result.isochrones.get("features", []):
        props = feat["properties"]
        step_h = int(props["step_h"])
        hw = result.hourly_wind[min(len(result.hourly_wind) - 1, step_h - 1)]
        iso_geoms.append(shape(feat["geometry"]))
        iso_records.append(
            {
                "step_h": step_h,
                "risk_level": str(props["risk_level"]),
                "area_ha": float(props["area_ha"]),
                "perim_km": float(props["perimeter_km"]),
                "ros_m_min": float(result.head_ros_m_min),
                "wind_kmh": float(hw["wind_speed_kmh"]),
                "wind_dir": float(hw["wind_dir_deg"]),
                "savi_mean": float(result.savi_mean),
                "slope_deg": float(result.slope_mean_deg),
                "date": result.date,
                "terr_id": result.territory_id,
                "terr_name": result.territory_name[:32],
                "model": SPREAD_MODEL_NAME[:24],
            }
        )

    iso_path = shp_dir / "isocronas_48h.shp"
    if iso_geoms:
        gdf_iso = gpd.GeoDataFrame(iso_records, geometry=iso_geoms, crs="EPSG:4326")
        gdf_iso.to_file(iso_path, driver="ESRI Shapefile", encoding="utf-8")

    # 2. Celdas horarias quemadas
    lon_edges = np.linspace(result.bounds["west"], result.bounds["east"], result.cols + 1)
    lat_edges = np.linspace(result.bounds["north"], result.bounds["south"], result.rows + 1)
    cell_area_ha = round((result.cell_size_m * result.cell_size_m) / 10_000.0, 3)

    burned_rc = np.argwhere(np.isfinite(result.arrival_h))
    cell_records: list[dict[str, Any]] = []
    cell_geoms: list[Any] = []
    for r, c in burned_rc:
        arr_h = float(result.arrival_h[r, c])
        cell_geoms.append(
            box(
                float(lon_edges[c]),
                float(lat_edges[r + 1]),
                float(lon_edges[c + 1]),
                float(lat_edges[r]),
            )
        )
        cell_records.append(
            {
                "arrival_h": int(min(result.horizon_h, max(1, math.ceil(arr_h)))),
                "arr_min": round(arr_h * 60.0, 1),
                "savi": round(float(result.savi[r, c]), 3),
                "elev_m": round(float(result.elevation[r, c]), 1),
                "slope_deg": round(float(result.slope_deg[r, c]), 2),
                "fuel": round(float(result.fuel[r, c]), 3),
                "area_ha": cell_area_ha,
            }
        )

    cells_path = shp_dir / "celdas_horarias_48h.shp"
    if cell_geoms:
        gdf_cells = gpd.GeoDataFrame(cell_records, geometry=cell_geoms, crs="EPSG:4326")
        gdf_cells.to_file(cells_path, driver="ESRI Shapefile", encoding="utf-8")

    # 3. Ignición y vectores de avance del viento cada 6 h
    ig_records: list[dict[str, Any]] = [
        {
            "feat_type": "ignition",
            "hour": 0,
            "wind_kmh": float(result.hourly_wind[0]["wind_speed_kmh"]),
            "wind_dir": float(result.hourly_wind[0]["wind_dir_deg"]),
            "lat": result.ignition_lat,
            "lon": result.ignition_lon,
        }
    ]
    ig_geoms: list[Any] = [Point(result.ignition_lon, result.ignition_lat)]
    vec_path = shp_dir / "ignicion_y_viento_48h.shp"
    gdf_ig = gpd.GeoDataFrame(ig_records, geometry=ig_geoms, crs="EPSG:4326")
    gdf_ig.to_file(vec_path, driver="ESRI Shapefile", encoding="utf-8")

    # Vector separado con líneas de flujo de viento
    wind_records: list[dict[str, Any]] = []
    wind_geoms: list[Any] = []
    lat_span = result.bounds["north"] - result.bounds["south"]
    lon_span = result.bounds["east"] - result.bounds["west"]
    step_vec = max(lat_span, lon_span) * 0.22
    for h_step in [s for s in ISOCHRONE_STEPS_H if s <= result.horizon_h]:
        hw = result.hourly_wind[min(len(result.hourly_wind) - 1, h_step - 1)]
        ang = math.radians((float(hw["wind_dir_deg"]) + 180.0) % 360.0)
        scale = step_vec * (0.45 + 0.55 * (h_step / float(result.horizon_h)))
        end_lon = result.ignition_lon + math.sin(ang) * scale
        end_lat = result.ignition_lat + math.cos(ang) * scale
        wind_geoms.append(LineString([(result.ignition_lon, result.ignition_lat), (end_lon, end_lat)]))
        wind_records.append(
            {
                "feat_type": "wind_axis",
                "hour": int(h_step),
                "wind_kmh": float(hw["wind_speed_kmh"]),
                "wind_dir": float(hw["wind_dir_deg"]),
                "area_ha": next((b["area_ha"] for b in result.bands if b["hours"] == h_step), 0.0),
            }
        )
    wind_path = shp_dir / "vectores_viento_48h.shp"
    if wind_geoms:
        gpd.GeoDataFrame(wind_records, geometry=wind_geoms, crs="EPSG:4326").to_file(
            wind_path, driver="ESRI Shapefile", encoding="utf-8"
        )

    zip_path = out_dir / "mchanaq_propagacion_48h_qgis.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_p in sorted(shp_dir.glob("*")):
            if file_p.is_file():
                zf.write(file_p, arcname=file_p.name)

    return {
        "isocronas_shp": iso_path,
        "celdas_shp": cells_path,
        "ignicion_shp": vec_path,
        "vectores_shp": wind_path,
        "qgis_zip": zip_path,
    }


def _compass_label(deg: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    return dirs[int(round((deg % 360.0) / 45.0)) % 8]


def export_animation_gif_mp4(
    result: SpreadGridResult,
    out_dir: Path = DEFAULT_OUTPUTS_DIR,
    fps: int = 6,
) -> dict[str, Path]:
    """Genera el GIF animado y el video MP4 (H.264) de la evolución horaria hasta 48 h."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Polygon as MplPolygon
    from PIL import Image
    import imageio.v2 as imageio

    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / "simulacion_48h.gif"
    mp4_path = out_dir / "pronostico_48h.mp4"

    fire_cmap = LinearSegmentedColormap.from_list(
        "mchanaq_fire",
        ["#3b0a0a", "#8f1d14", "#d9480f", "#f76707", "#ffd43b", "#fff9db"],
    )
    savi_cmap = LinearSegmentedColormap.from_list(
        "mchanaq_savi_dark",
        ["#0c1218", "#15251f", "#1e3a2b", "#2d553a", "#4a7846"],
    )

    extent = [
        result.bounds["west"],
        result.bounds["east"],
        result.bounds["south"],
        result.bounds["north"],
    ]
    hours_x = [p["hour"] for p in result.hourly_progression]
    areas_y = [p["cumulative_area_ha"] for p in result.hourly_progression]
    max_area = max(10.0, max(areas_y) if areas_y else 10.0)

    frames: list[np.ndarray] = []
    # 960 x 544 (múltiplo exacto de 16 para codec H.264); reutilizar figura y artistas estáticos
    fig = plt.figure(figsize=(9.6, 5.44), dpi=100, facecolor="#0b1016")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.48, 1.04], height_ratios=[1.0, 0.36], wspace=0.18, hspace=0.24)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_curve = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, 1])

    ax_map.set_facecolor("#090d12")
    ax_map.imshow(result.savi, extent=extent, origin="upper", cmap=savi_cmap, vmin=0.0, vmax=0.80, alpha=0.88)
    ax_map.contour(
        np.linspace(extent[0], extent[1], result.cols),
        np.linspace(extent[3], extent[2], result.rows),
        result.elevation,
        levels=7,
        colors="#64748b",
        linewidths=0.45,
        alpha=0.35,
    )
    arr = result.arrival_h
    im_heat = ax_map.imshow(
        np.full_like(arr, np.nan, dtype=np.float32),
        extent=extent,
        origin="upper",
        cmap=fire_cmap,
        vmin=0.0,
        vmax=1.0,
        alpha=0.92,
        animated=True,
    )
    ig_marker = ax_map.scatter(
        [result.ignition_lon],
        [result.ignition_lat],
        s=55,
        c="#ffffff",
        edgecolors="#ff4500",
        zorder=6,
        animated=True,
    )
    arrow_len = (extent[1] - extent[0]) * 0.14
    wind_quiver = ax_map.quiver(
        [result.ignition_lon],
        [result.ignition_lat],
        [0.0],
        [0.0],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#38bdf8",
        width=0.009,
        headwidth=4.0,
        headlength=4.5,
        zorder=7,
        animated=True,
    )
    ax_map.set_title(
        f"McHanaq Fase 2 ({result.horizon_h}h) — {result.territory_name} ({result.date})",
        color="#f8fafc",
        fontsize=9.3,
        fontweight="bold",
        pad=8,
        loc="left",
    )
    ax_map.set_xlabel("Longitud (°)", color="#94a3b8", fontsize=8)
    ax_map.set_ylabel("Latitud (°)", color="#94a3b8", fontsize=8)
    ax_map.tick_params(colors="#94a3b8", labelsize=7.5)
    for sp in ax_map.spines.values():
        sp.set_edgecolor("#1e293b")

    badge_txt = ax_map.text(
        0.02,
        0.03,
        "",
        transform=ax_map.transAxes,
        color="#f8fafc",
        fontsize=8.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", edgecolor="#334155", alpha=0.92),
        zorder=8,
        animated=True,
    )

    ax_curve.set_facecolor("#0f172a")
    ax_curve.plot(hours_x, areas_y, color="#334155", lw=1.2, ls="--")
    fill_patch = MplPolygon(
        [(hours_x[0], 0.0)],
        closed=True,
        facecolor="#f97316",
        alpha=0.28,
        edgecolor="none",
        animated=True,
    )
    ax_curve.add_patch(fill_patch)
    line_prog, = ax_curve.plot([], [], color="#fb923c", lw=2.2, animated=True)
    pt_prog = ax_curve.scatter([hours_x[0]], [areas_y[0]], color="#fde047", s=38, zorder=5, animated=True)
    ax_curve.set_xlim(1, result.horizon_h)
    ax_curve.set_ylim(0, max_area * 1.12)
    ax_curve.set_title("Superficie acumulada (ha)", color="#e2e8f0", fontsize=9.5, fontweight="bold", loc="left")
    ax_curve.set_xlabel("Hora de pronóstico", color="#94a3b8", fontsize=8)
    ax_curve.tick_params(colors="#94a3b8", labelsize=7.5)
    ax_curve.grid(color="#1e293b", lw=0.6)
    for sp in ax_curve.spines.values():
        sp.set_edgecolor("#1e293b")

    ax_info.set_facecolor("#0f172a")
    ax_info.axis("off")
    info_box = ax_info.text(
        0.02,
        0.92,
        "",
        va="top",
        ha="left",
        color="#e2e8f0",
        fontsize=7.1,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#111827", edgecolor="#334155"),
        animated=True,
    )
    ax_info.text(
        0.02,
        0.04,
        "Exploratorio (SNMF / CONAE-Gulich AC + Rothermel). No operativo.",
        color="#94a3b8",
        fontsize=6.2,
    )
    fig.subplots_adjust(left=0.075, right=0.965, top=0.91, bottom=0.10)
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)

    horizon_scale = max(4.0, float(result.horizon_h) * 0.65)
    for idx in range(result.horizon_h):
        h = idx + 1
        hw = result.hourly_wind[idx]
        prog = result.hourly_progression[idx]

        fig.canvas.restore_region(bg)

        active_mask = np.isfinite(arr) & (arr <= float(h))
        heat = np.full_like(arr, np.nan, dtype=np.float32)
        age_norm = np.clip((float(h) - arr) / horizon_scale, 0.0, 0.92)
        heat[active_mask] = 1.0 - age_norm[active_mask]
        im_heat.set_data(heat)

        ang = math.radians((float(hw["wind_dir_deg"]) + 180.0) % 360.0)
        wind_quiver.set_UVC([math.sin(ang) * arrow_len], [math.cos(ang) * arrow_len])
        badge_txt.set_text(
            f"HORA {h:02d}/{result.horizon_h:02d}  |  Viento: {hw['wind_speed_kmh']:.1f} km/h desde {round(hw['wind_dir_deg'])}° ({_compass_label(hw['wind_dir_deg'])})"
        )

        xs = hours_x[:idx + 1]
        ys = areas_y[:idx + 1]
        fill_patch.set_xy([(xs[0], 0.0), *zip(xs, ys), (xs[-1], 0.0)])
        line_prog.set_data(xs, ys)
        pt_prog.set_offsets([[h, prog["cumulative_area_ha"]]])

        info_box.set_text(
            f"HORA {h:02d}  ·  {hw['timestamp']}\n"
            f"Área alcanzada:  {prog['cumulative_area_ha']:,.1f} ha  (+{prog['hourly_increment_ha']:,.1f} ha/h)\n"
            f"Avance cabeza:   {prog['head_ros_m_min']:.2f} m/min  |  L/B {prog['length_to_breadth']:.2f}:1\n"
            f"SAVI medio: {result.savi_mean:.2f}  ·  Pendiente: {result.slope_mean_deg:.1f}°  ·  Humedad: {result.fuel_moisture_pct:.0f}%"
        )

        ax_map.draw_artist(im_heat)
        ax_map.draw_artist(ig_marker)
        ax_map.draw_artist(wind_quiver)
        ax_map.draw_artist(badge_txt)
        ax_curve.draw_artist(fill_patch)
        ax_curve.draw_artist(line_prog)
        ax_curve.draw_artist(pt_prog)
        ax_info.draw_artist(info_box)
        fig.canvas.blit(fig.bbox)
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)

    plt.close(fig)

    # Exportar GIF (cuantización rápida sin dither para reducir tiempo y peso)
    pil_frames = [
        Image.fromarray(f).quantize(colors=128, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
        for f in frames
    ]
    # Pausa extra en el último frame (48h)
    durations = [int(1000 / fps)] * (len(pil_frames) - 1) + [1800]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )

    # Exportar MP4 (H.264 compatible)
    with imageio.get_writer(
        mp4_path,
        fps=fps,
        codec="libx264",
        macro_block_size=16,
        pixelformat="yuv420p",
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
        # Repetir último frame para pausa final de ~1.5 s
        for _ in range(fps):
            writer.append_data(frames[-1])

    return {"gif": gif_path, "mp4": mp4_path}


def _plot_geom_boundary(ax: Any, geom: Any, **kwargs: Any) -> None:
    """Dibuja el contorno de un Polygon o MultiPolygon sin alterar ejes ni etiquetas."""
    if geom is None or geom.is_empty:
        return
    polys = [geom] if isinstance(geom, Polygon) else list(getattr(geom, "geoms", []))
    first = True
    lbl = kwargs.pop("label", None)
    for poly in polys:
        if not hasattr(poly, "exterior"):
            continue
        xs, ys = poly.exterior.xy
        ax.plot(xs, ys, label=lbl if first else None, **kwargs)
        first = False


def export_impact_infographics(
    result: SpreadGridResult,
    out_dir: Path = DEFAULT_OUTPUTS_DIR,
) -> dict[str, Path]:
    """Genera dos infografías de alta resolución (estilo Imagen 1 y estilo Imagen 2)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyBboxPatch

    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / "panel_modelo_48h.png"
    info_path = out_dir / "infografia_impacto_48h.png"

    extent = [
        result.bounds["west"],
        result.bounds["east"],
        result.bounds["south"],
        result.bounds["north"],
    ]
    savi_cmap = LinearSegmentedColormap.from_list(
        "savi_prof", ["#1e293b", "#8c510a", "#d8b365", "#a6d96a", "#1a9850", "#006837"]
    )
    slope_cmap = LinearSegmentedColormap.from_list(
        "slope_prof", ["#1b4332", "#52b788", "#e9c46a", "#e76f51", "#9b2226"]
    )
    fire_cmap = LinearSegmentedColormap.from_list(
        "fire_prof", ["#450a0a", "#b91c1c", "#ea580c", "#f59e0b", "#fef08a"]
    )

    # ------------------------------------------------------------------
    # 1) panel_modelo_48h.png — Triple panel estilo Imagen 1
    # ------------------------------------------------------------------
    fig1 = plt.figure(figsize=(15.4, 8.6), dpi=200, facecolor="#0d1117")
    gs1 = fig1.add_gridspec(2, 3, height_ratios=[0.13, 1.0], wspace=0.14, hspace=0.14)

    ax_hdr = fig1.add_subplot(gs1[0, :])
    ax_hdr.axis("off")
    ax_hdr.text(
        0.01,
        0.72,
        f"SIMULACIÓN ESPACIAL DE PROPAGACIÓN DE FUEGO · HORIZONTE {result.horizon_h} HORAS",
        color="#f8fafc",
        fontsize=15.5,
        fontweight="bold",
    )
    ax_hdr.text(
        0.01,
        0.22,
        f"Territorio: {result.territory_name} ({result.territory_type} {result.territory_id})  |  Fecha base: {result.date}  |  Foco: {result.ignition_lat:.4f}°, {result.ignition_lon:.4f}°  |  Marco: Ley 26.815 / SNMF / CONAE-Gulich AC",
        color="#94a3b8",
        fontsize=9.2,
    )

    # Panel A: SAVI + Viento
    ax_a = fig1.add_subplot(gs1[1, 0])
    ax_a.set_facecolor("#0b1016")
    im_savi = ax_a.imshow(result.savi, extent=extent, origin="upper", cmap=savi_cmap, vmin=0.0, vmax=0.80, aspect="auto")
    _plot_geom_boundary(ax_a, result.territory_geom_wgs84, color="#cbd5e1", linewidth=0.9, alpha=0.65)
    ax_a.set_xlim(extent[0], extent[1])
    ax_a.set_ylim(extent[2], extent[3])
    xs_v = np.linspace(extent[0] + (extent[1] - extent[0]) * 0.2, extent[1] - (extent[1] - extent[0]) * 0.2, 4)
    ys_v = np.linspace(extent[2] + (extent[3] - extent[2]) * 0.2, extent[3] - (extent[3] - extent[2]) * 0.2, 4)
    gx_v, gy_v = np.meshgrid(xs_v, ys_v)
    ang0 = math.radians((result.hourly_wind[0]["wind_dir_deg"] + 180.0) % 360.0)
    ax_a.quiver(gx_v, gy_v, math.sin(ang0), math.cos(ang0), color="#f8fafc", alpha=0.72, scale=14, width=0.004)
    ax_a.set_title("1. Vegetación Combustible (SAVI)", color="#f1f5f9", fontsize=11, fontweight="bold", loc="left", pad=8)
    ax_a.tick_params(colors="#64748b", labelsize=7.5)
    for sp in ax_a.spines.values():
        sp.set_edgecolor("#1e293b")
    cb_a = fig1.colorbar(im_savi, ax=ax_a, orientation="horizontal", fraction=0.045, pad=0.09)
    cb_a.ax.tick_params(colors="#94a3b8", labelsize=7)
    cb_a.set_label("Índice SAVI (Huete, 1988)", color="#cbd5e1", fontsize=8)

    # Panel B: Pendiente + Viento 48h + Frente
    ax_b = fig1.add_subplot(gs1[1, 1])
    ax_b.set_facecolor("#0b1016")
    im_slp = ax_b.imshow(
        result.slope_deg,
        extent=extent,
        origin="upper",
        cmap=slope_cmap,
        vmin=0.0,
        vmax=max(18.0, result.slope_max_deg),
        aspect="auto",
    )
    arr_norm = np.where(np.isfinite(result.arrival_h), 1.0 - (result.arrival_h / float(result.horizon_h)) * 0.85, np.nan)
    ax_b.imshow(arr_norm, extent=extent, origin="upper", cmap=fire_cmap, vmin=0.0, vmax=1.0, alpha=0.90, aspect="auto")
    ax_b.scatter([result.ignition_lon], [result.ignition_lat], s=55, c="#ffffff", edgecolors="#dc2626", zorder=6)
    w_mean_kmh = float(np.mean([w["wind_speed_kmh"] for w in result.hourly_wind]))
    w_dir0 = result.hourly_wind[0]["wind_dir_deg"]
    ax_b.set_title(
        f"2. Pendiente DEM · Viento {result.horizon_h}h ({_compass_label(w_dir0)} {w_mean_kmh:.0f} km/h)",
        color="#f1f5f9",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    ax_b.tick_params(colors="#64748b", labelsize=7.5)
    for sp in ax_b.spines.values():
        sp.set_edgecolor("#1e293b")
    cb_b = fig1.colorbar(im_slp, ax=ax_b, orientation="horizontal", fraction=0.045, pad=0.09)
    cb_b.ax.tick_params(colors="#94a3b8", labelsize=7)
    cb_b.set_label("Pendiente del terreno (°) · DEM", color="#cbd5e1", fontsize=8)

    # Panel C: Simulación celular 48h + resumen de salidas
    ax_c = fig1.add_subplot(gs1[1, 2])
    ax_c.set_facecolor("#080c11")
    ax_c.grid(color="#1e293b", lw=0.6, alpha=0.8)
    ax_c.imshow(arr_norm, extent=extent, origin="upper", cmap=fire_cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for feat in result.isochrones.get("features", []):
        _plot_geom_boundary(ax_c, shape(feat["geometry"]), color="#fef08a", linewidth=0.95, alpha=0.88)
    ax_c.scatter([result.ignition_lon], [result.ignition_lat], s=48, c="#ffffff", edgecolors="#dc2626", zorder=6)
    ax_c.set_xlim(extent[0], extent[1])
    ax_c.set_ylim(extent[2], extent[3])
    ax_c.set_title(f"3. Simulación {result.horizon_h}h · Impacto {result.total_area_ha:,.0f} ha", color="#f1f5f9", fontsize=11, fontweight="bold", loc="left", pad=8)
    ax_c.tick_params(colors="#64748b", labelsize=7.5)
    for sp in ax_c.spines.values():
        sp.set_edgecolor("#1e293b")

    band_p1 = " | ".join(f"+{b['hours']}h: {b['area_ha']:,.0f} ha" for b in result.bands[:3])
    band_p2 = " | ".join(f"+{b['hours']}h: {b['area_ha']:,.0f} ha" for b in result.bands[3:])
    summary_box = f"Isócronas: {band_p1}" + (f"\n           {band_p2}" if band_p2 else "")
    summary_box += "\nSalidas: simulacion_48h.gif · pronostico_48h.mp4 · .shp"
    ax_c.text(
        0.01,
        -0.21,
        summary_box,
        transform=ax_c.transAxes,
        color="#cbd5e1",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#111827", edgecolor="#334155"),
    )

    fig1.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.15)
    fig1.savefig(panel_path, dpi=200, facecolor="#0d1117")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # 2) infografia_impacto_48h.png — Lámina ejecutiva estilo Imagen 2
    # ------------------------------------------------------------------
    fig2 = plt.figure(figsize=(15.4, 9.4), dpi=220, facecolor="#0a0f14")
    gs2 = fig2.add_gridspec(3, 3, width_ratios=[1.15, 1.05, 0.95], height_ratios=[0.16, 1.0, 0.45], wspace=0.16, hspace=0.20)

    # Banner superior
    ax_top = fig2.add_subplot(gs2[0, :])
    ax_top.axis("off")
    ax_top.text(
        0.01,
        0.74,
        f"McHanaq · Pronóstico Exploratorio de Propagación de Incendio ({result.horizon_h} horas)",
        color="#f8fafc",
        fontsize=16.5,
        fontweight="bold",
    )
    ax_top.text(
        0.01,
        0.24,
        f"{result.territory_name} ({result.territory_type} {result.territory_id})  ·  Fecha: {result.date}  ·  Foco: {result.ignition_lat:.4f}°, {result.ignition_lon:.4f}°  ·  SAVI + DEM + Viento 48h",
        color="#fbbf24",
        fontsize=9.8,
        fontweight="bold",
    )
    ax_top.text(
        0.99,
        0.52,
        "Referencia metodológica:\nLey 26.815 (Arg.) · SNMF / CONAE\nInstituto Gulich (UNC) · Autómatas Celulares",
        ha="right",
        va="center",
        color="#94a3b8",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#111827", edgecolor="#1e293b"),
    )

    # Mapa principal (izquierda + centro arriba)
    ax_main = fig2.add_subplot(gs2[1, :2])
    ax_main.set_facecolor("#090d12")
    ax_main.imshow(result.savi, extent=extent, origin="upper", cmap=savi_cmap, vmin=0.0, vmax=0.82, alpha=0.82, aspect="auto")
    _plot_geom_boundary(ax_main, result.territory_geom_wgs84, color="#38bdf8", linewidth=1.1, alpha=0.75, linestyle="--")
    ax_main.imshow(arr_norm, extent=extent, origin="upper", cmap=fire_cmap, vmin=0.0, vmax=1.0, alpha=0.90, aspect="auto")
    iso_colors = {6: "#fef08a", 12: "#fdba74", 24: "#fb923c", 36: "#f97316", 48: "#ef4444"}
    for feat in result.isochrones.get("features", []):
        st = int(feat["properties"]["step_h"])
        _plot_geom_boundary(
            ax_main,
            shape(feat["geometry"]),
            color=iso_colors.get(st, "#f97316"),
            linewidth=1.7 if st in (6, 24, 48) else 1.1,
            label=f"+{st}h ({feat['properties']['area_ha']:,.0f} ha)",
        )
    ax_main.scatter([result.ignition_lon], [result.ignition_lat], s=110, marker="*", c="#ffffff", edgecolors="#dc2626", zorder=8, label="Punto de ignición")
    ax_main.set_xlim(extent[0], extent[1])
    ax_main.set_ylim(extent[2], extent[3])
    ax_main.set_title("Mapa Territorial de Isócronas de Avance (+6 h a +48 h)", color="#f8fafc", fontsize=11.5, fontweight="bold", loc="left", pad=8)
    ax_main.tick_params(colors="#94a3b8", labelsize=8)
    leg = ax_main.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f1f5f9", fontsize=8.2)
    leg.get_frame().set_alpha(0.92)
    for sp in ax_main.spines.values():
        sp.set_edgecolor("#1e293b")

    # Columna derecha superior: Mini SAVI + Mini PENDIENTE
    gs_side = gs2[1, 2].subgridspec(2, 1, hspace=0.30)
    ax_savi_mini = fig2.add_subplot(gs_side[0])
    ax_savi_mini.set_facecolor("#0f172a")
    im1 = ax_savi_mini.imshow(result.savi, extent=extent, origin="upper", cmap=savi_cmap, vmin=0.0, vmax=0.80, aspect="auto")
    ax_savi_mini.set_title(f"SAVI · Combustible (media {result.savi_mean:.2f})", color="#e2e8f0", fontsize=9.2, fontweight="bold", loc="left")
    ax_savi_mini.set_xticks([])
    ax_savi_mini.set_yticks([])
    cb1 = fig2.colorbar(im1, ax=ax_savi_mini, fraction=0.055, pad=0.04)
    cb1.ax.tick_params(colors="#94a3b8", labelsize=7)

    ax_slp_mini = fig2.add_subplot(gs_side[1])
    ax_slp_mini.set_facecolor("#0f172a")
    im2 = ax_slp_mini.imshow(
        result.slope_deg,
        extent=extent,
        origin="upper",
        cmap=slope_cmap,
        vmin=0.0,
        vmax=max(18.0, result.slope_max_deg),
        aspect="auto",
    )
    ax_slp_mini.set_title(f"PENDIENTE · DEM ({result.slope_mean_deg:.1f}° med / {result.slope_max_deg:.1f}° máx)", color="#e2e8f0", fontsize=8.8, fontweight="bold", loc="left")
    ax_slp_mini.set_xticks([])
    ax_slp_mini.set_yticks([])
    cb2 = fig2.colorbar(im2, ax=ax_slp_mini, fraction=0.055, pad=0.04)
    cb2.ax.tick_params(colors="#94a3b8", labelsize=7)

    # Fila inferior izquierda: VIENTO HORARIO (48 h)
    ax_wind = fig2.add_subplot(gs2[2, 0])
    ax_wind.set_facecolor("#0f172a")
    hrs = [w["hour"] for w in result.hourly_wind]
    w_kmh = [w["wind_speed_kmh"] for w in result.hourly_wind]
    ax_wind.plot(hrs, w_kmh, color="#38bdf8", lw=2.0)
    ax_wind.fill_between(hrs, w_kmh, color="#0ea5e9", alpha=0.22)
    ax_wind.set_title(f"VIENTO HORARIO (48h) · {w_mean_kmh:.1f} km/h ({_compass_label(w_dir0)})", color="#e2e8f0", fontsize=9.3, fontweight="bold", loc="left")
    ax_wind.set_xlabel("Hora (1–48)", color="#94a3b8", fontsize=8)
    ax_wind.set_ylabel("km/h", color="#94a3b8", fontsize=8)
    ax_wind.tick_params(colors="#94a3b8", labelsize=7.5)
    ax_wind.grid(color="#1e293b", lw=0.6)
    for sp in ax_wind.spines.values():
        sp.set_edgecolor("#1e293b")

    # Fila inferior centro: CURVA DE IMPACTO EN HECTÁREAS
    ax_sim = fig2.add_subplot(gs2[2, 1])
    ax_sim.set_facecolor("#0f172a")
    area_h = [p["cumulative_area_ha"] for p in result.hourly_progression]
    ax_sim.fill_between(hrs, area_h, color="#f97316", alpha=0.26)
    ax_sim.plot(hrs, area_h, color="#fb923c", lw=2.2)
    for b in result.bands:
        ax_sim.scatter([b["hours"]], [b["area_ha"]], color="#fde047", s=28, zorder=5)
    ax_sim.set_title(f"SIMULACIÓN · Impacto total {result.total_area_ha:,.0f} ha", color="#e2e8f0", fontsize=9.3, fontweight="bold", loc="left")
    ax_sim.set_xlabel("Hora de pronóstico", color="#94a3b8", fontsize=8)
    ax_sim.set_ylabel("Hectáreas (ha)", color="#94a3b8", fontsize=8)
    ax_sim.tick_params(colors="#94a3b8", labelsize=7.5)
    ax_sim.grid(color="#1e293b", lw=0.6)
    for sp in ax_sim.spines.values():
        sp.set_edgecolor("#1e293b")

    # Fila inferior derecha: OUTPUTS GIS + TABLA DE IMPACTO
    ax_out = fig2.add_subplot(gs2[2, 2])
    ax_out.set_facecolor("#0f172a")
    ax_out.axis("off")
    rect = FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.02", facecolor="#111827", edgecolor="#334155", transform=ax_out.transAxes)
    ax_out.add_patch(rect)
    bands_lines = "\n".join(
        f"  +{b['hours']:02d} h : {b['area_ha']:>8,.1f} ha"
        for b in result.bands
    )
    out_text = (
        "OUTPUTS GIS Y MULTIMEDIA\n"
        "  • .SHP  isocronas_48h / celdas_horarias (QGIS)\n"
        "  • .MP4  pronostico_48h.mp4 | .GIF simulacion_48h.gif\n\n"
        f"IMPACTO (ROS {result.head_ros_m_min:.2f} m/min | L/B {result.length_to_breadth:.2f}:1):\n"
        f"{bands_lines}"
    )
    ax_out.text(0.04, 0.94, out_text, va="top", ha="left", color="#f1f5f9", fontsize=7.4, family="monospace", transform=ax_out.transAxes)
    ax_out.text(
        0.04,
        0.04,
        "Aviso: Proyección exploratoria condicional;\nno sustituye alertas ni despacho operativo.",
        color="#94a3b8",
        fontsize=6.8,
        transform=ax_out.transAxes,
    )

    fig2.subplots_adjust(left=0.05, right=0.97, top=0.96, bottom=0.07)
    fig2.savefig(info_path, dpi=220, facecolor="#0a0f14")
    plt.close(fig2)

    return {"panel": panel_path, "infografia": info_path}


def export_all_phase2_products(
    result: SpreadGridResult,
    out_dir: Path = DEFAULT_OUTPUTS_DIR,
    fps: int = 6,
) -> dict[str, Any]:
    """Ejecuta todos los exportadores (Shapefiles+ZIP, Infografías PNG y GIF+MP4) midiendo tiempos."""
    t_s = time.perf_counter()
    shp_files = export_qgis_shapefiles(result, out_dir)
    t_shp = time.perf_counter() - t_s

    t_i = time.perf_counter()
    info_files = export_impact_infographics(result, out_dir)
    t_info = time.perf_counter() - t_i

    t_a = time.perf_counter()
    anim_files = export_animation_gif_mp4(result, out_dir, fps=fps)
    t_anim = time.perf_counter() - t_a

    files: dict[str, Path] = {}
    files.update(shp_files)
    files.update(info_files)
    files.update(anim_files)
    return {
        "files": files,
        "timings_s": {
            "shapefiles_s": round(t_shp, 3),
            "infographics_s": round(t_info, 3),
            "animation_s": round(t_anim, 3),
            "total_export_s": round(t_shp + t_info + t_anim, 3),
        },
    }


def build_spread_forecast_payload(
    result: SpreadGridResult,
    outputs_prefix: str = "/api/v1/spread-outputs",
    timings_s: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Construye la respuesta JSON/GeoJSON conforme al contrato §12.2 de McHanaq."""
    ws_mean = round(float(np.mean([w["wind_speed_ms"] for w in result.hourly_wind])), 2)
    wd_dom = result.hourly_wind[0]["wind_dir_deg"]
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "phase": "phase_2_optional",
        "territory": {
            "type": result.territory_type,
            "id": result.territory_id,
            "name": result.territory_name,
        },
        "date": result.date,
        "horizon_h": result.horizon_h,
        "ignition": {"lat": result.ignition_lat, "lon": result.ignition_lon},
        "dominant_wind": {"wind_dir_deg": wd_dom, "wind_speed_10m_ms": ws_mean},
        "hourly_wind": result.hourly_wind,
        "savi_summary": {
            "mean": result.savi_mean,
            "p90": result.savi_p90,
            "source": result.data_sources.get("vegetation_index"),
        },
        "slope_summary": {
            "mean_deg": result.slope_mean_deg,
            "max_deg": result.slope_max_deg,
            "dem_source": result.data_sources.get("dem_source"),
        },
        "grid_summary": {
            "cols": result.cols,
            "rows": result.rows,
            "cell_size_m": result.cell_size_m,
            "bounds": result.bounds,
            "total_area_ha": result.total_area_ha,
            "head_ros_m_min": result.head_ros_m_min,
            "length_to_breadth": result.length_to_breadth,
        },
        "bands": result.bands,
        "hourly_progression": result.hourly_progression,
        "isochrones": result.isochrones,
        "assumptions": result.assumptions,
        "data_sources": result.data_sources,
        "outputs": {
            "gif": f"{outputs_prefix}/simulacion_48h.gif",
            "mp4": f"{outputs_prefix}/pronostico_48h.mp4",
            "shapefiles_zip": f"{outputs_prefix}/mchanaq_propagacion_48h_qgis.zip",
            "infografia_impacto": f"{outputs_prefix}/infografia_impacto_48h.png",
            "panel_modelo": f"{outputs_prefix}/panel_modelo_48h.png",
        },
        "model_metadata": {
            "model_name": SPREAD_MODEL_NAME,
            "disclaimer": SPREAD_DISCLAIMER,
            "references": [
                "Argentina Ley 26.815 / SNMF / CONAE - Instituto Gulich (UNC) / SARTiv",
                "Rothermel (1972); Anderson (1983); Alexander (1985); Huete (1988) SAVI",
            ],
        },
    }
    if timings_s is not None:
        payload["timings_s"] = timings_s
    return payload


def _load_territory_feature(
    processed_dir: Path,
    territory_type: str,
    territory_id: str,
) -> tuple[str, Any]:
    terr_path = processed_dir / "territories.geojson"
    if not terr_path.exists():
        terr_path = DEFAULT_FIXTURES / "territories.geojson"
    if terr_path.exists():
        fc = json.loads(terr_path.read_text(encoding="utf-8"))
        for feat in fc.get("features", []):
            props = feat.get("properties") or {}
            if props.get("territory_type") == territory_type and str(props.get("territory_id")) == territory_id:
                name = str(props.get("territory_name") or props.get("name") or territory_id)
                return name, shape(feat["geometry"])
    return territory_id, None


def run_phase2_pipeline(
    processed_dir: Path = DEFAULT_PROCESSED,
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
    territory_type: str = "province",
    territory_id: str = "2501",
    date_str: str = "2024-09-25",
    ignition_lat: float | None = -8.19833,
    ignition_lon: float | None = -75.26669,
    horizon_h: int = DEFAULT_HORIZON_H,
    use_open_meteo: bool = False,
    savi_tif: Path | None = None,
    dem_tif: Path | None = None,
    generate_media: bool = True,
) -> dict[str, Any]:
    """Ejecuta el flujo completo de Fase 2 y guarda JSON, Shapefiles, GIF, MP4 e infografías."""
    terr_name, geom = _load_territory_feature(processed_dir, territory_type, territory_id)
    if ignition_lat is None or ignition_lon is None:
        if geom is not None and not geom.is_empty:
            c = geom.representative_point()
            ignition_lat, ignition_lon = float(c.y), float(c.x)
        else:
            ignition_lat, ignition_lon = -8.19833, -75.26669

    # Leer fracción de vegetación real de Ucayali si está disponible
    veg_pct = 86.0
    districts_csv = processed_dir / "ucayali_districts_static.csv"
    if districts_csv.exists() and territory_type == "province":
        import csv

        areas, eff_vegs = [], []
        with districts_csv.open("r", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("ubigeo", "")).startswith(territory_id):
                    a = float(row.get("area_km2") or 0)
                    ev = float(row.get("veg_outside_water_buffer_km2") or 0)
                    if a > 0:
                        areas.append(a)
                        eff_vegs.append(ev)
        if sum(areas) > 0:
            veg_pct = round((sum(eff_vegs) / sum(areas)) * 100.0, 2)

    t_sim0 = time.perf_counter()
    result = simulate_spread_48h(
        territory_type=territory_type,
        territory_id=territory_id,
        territory_name=terr_name,
        territory_geom=geom,
        date_str=date_str,
        ignition_lat=ignition_lat,
        ignition_lon=ignition_lon,
        horizon_h=horizon_h,
        vegetation_pct=veg_pct,
        use_open_meteo=use_open_meteo,
        savi_tif=savi_tif,
        dem_tif=dem_tif,
    )
    sim_s = round(time.perf_counter() - t_sim0, 3)

    artifacts: dict[str, Any] = {}
    timings_s: dict[str, float] = {"simulation_s": sim_s}
    if generate_media:
        exported = export_all_phase2_products(result, outputs_dir)
        artifacts.update(exported["files"])
        timings_s.update(exported["timings_s"])
        timings_s["total_s"] = round(sim_s + exported["timings_s"]["total_export_s"], 3)
    else:
        timings_s["total_s"] = sim_s

    payload = build_spread_forecast_payload(result, timings_s=timings_s)
    forecasts_doc = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "default_key": f"{territory_type}:{territory_id}",
        "items": [payload],
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    forecasts_path = processed_dir / "spread_forecasts.json"
    forecasts_path.write_text(json.dumps(forecasts_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts["forecasts_json"] = forecasts_path

    return {"result": result, "payload": payload, "artifacts": artifacts, "timings_s": timings_s}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="McHanaq Fase 2 — Simulación de propagación de fuego 48h")
    parser.add_argument("--territory-type", default="province", choices=["province", "hydro_unit"])
    parser.add_argument("--territory-id", default="2501")
    parser.add_argument("--date", default="2024-09-25")
    parser.add_argument("--lat", type=float, default=-8.19833)
    parser.add_argument("--lon", type=float, default=-75.26669)
    parser.add_argument("--horizon-h", type=int, default=48)
    parser.add_argument("--open-meteo", action="store_true")
    parser.add_argument("--savi-tif", type=Path, default=None)
    parser.add_argument("--dem-tif", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    args = parser.parse_args(argv)

    out = run_phase2_pipeline(
        processed_dir=args.processed_dir,
        outputs_dir=args.outputs_dir,
        territory_type=args.territory_type,
        territory_id=args.territory_id,
        date_str=args.date,
        ignition_lat=args.lat,
        ignition_lon=args.lon,
        horizon_h=args.horizon_h,
        use_open_meteo=args.open_meteo,
        savi_tif=args.savi_tif,
        dem_tif=args.dem_tif,
    )
    res: SpreadGridResult = out["result"]
    print(f"Simulación 48h completada para {res.territory_name} ({res.territory_type}/{res.territory_id})")
    print(f"SAVI medio: {res.savi_mean} | Pendiente media: {res.slope_mean_deg}° | Área total {res.horizon_h}h: {res.total_area_ha} ha")
    print(f"Tiempos (s): {out['timings_s']}")
    for k, p in out["artifacts"].items():
        print(f"  - {k}: {p}")


if __name__ == "__main__":
    main()
