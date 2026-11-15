"""McHanaq API — Backend de solo lectura (Fase 1, resolución diaria por defecto)."""

import csv
import io
import json
import math
import os
import time
from datetime import date as dt_date
from pathlib import Path

import httpx
from fastapi import Body, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.getenv("MCHANAQ_DATA_DIR", ROOT / "data" / "processed"))
if not DATA.is_absolute():
    DATA = ROOT / DATA
NO_CACHE = os.getenv("MCHANAQ_NO_CACHE", "0") == "1"
CORS_ORIGINS = [
    o.strip().rstrip("/")
    for o in os.getenv("MCHANAQ_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]
CORS_ORIGIN_REGEX = os.getenv("MCHANAQ_CORS_ORIGIN_REGEX", "").strip() or None

TYPES = {"province", "hydro_unit"}
SENSORS = {"MODIS_TERRA", "MODIS_AQUA", "VIIRS_SNPP"}
ACTIVITY_METRICS = {
    "activity_rate",
    "fire_pixels_nominal_high",
    "fire_pixels_low",
    "observation_coverage",
    "fire_pixel_days_8_9",
    "active_days",
    "max_consecutive_active_days",
}
MONTHLY_SUMMARY_METRICS = {
    "fire_pixel_days_8_9",
    "active_days",
    "max_consecutive_active_days",
}
ENV_METRICS = {
    "t_amb_c",
    "relative_humidity_2m_pct",
    "t_suelo_c",
    "lst_valid_pixel_pct",
    "wind_dir_deg",
    "wind_speed_10m_ms",
    "elevation_m",
    "conditions_classification",
}


FIXTURES_DATA = ROOT / "data" / "fixtures"


def load_json(name: str):
    """Carga un JSON/GeoJSON desde DATA si existe, con respaldo en data/fixtures."""
    path = DATA / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    fallback = FIXTURES_DATA / name
    return json.loads(fallback.read_text(encoding="utf-8")) if fallback.exists() else None


META = load_json("metadata.json") or {}
TERR = (
    load_json("peru_territories.geojson")
    or load_json("territories.geojson")
    or {"type": "FeatureCollection", "features": []}
)
DAILY = (load_json("activity_daily.json") or {}).get("items", [])
SUMMARY = (load_json("activity_summary.json") or {}).get("items", [])
COMPARE = (load_json("sensor_comparison.json") or {}).get("items", [])
ASSESS = (load_json("conditions_assessment.json") or {}).get("items", [])
HOTSPOTS = load_json("hotspots.geojson")


class ApiError(Exception):
    """Error de dominio con formato uniforme para el cliente."""

    def __init__(self, status: int, code: str, detail: str):
        self.status = status
        self.code = code
        self.detail = detail


app = FastAPI(title="McHanaq API", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_cache_header(request: Request, call_next):
    response = await call_next(request)
    if request.method == "GET" and response.status_code == 200 and not request.url.path.endswith("/health"):
        response.headers["Cache-Control"] = "no-store" if NO_CACHE else "public, max-age=300"
    return response


@app.exception_handler(ApiError)
async def api_error_handler(_: Request, exc: ApiError):
    return JSONResponse(status_code=exc.status, content={"code": exc.code, "detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, __: RequestValidationError):
    return JSONResponse(status_code=400, content={"code": "INVALID_PARAMETER", "detail": "Parámetros inválidos"})


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, __: Exception):
    return JSONResponse(status_code=500, content={"code": "INTERNAL_ERROR", "detail": "Error interno del servidor"})


def check_type(t: str) -> None:
    if t not in TYPES:
        raise ApiError(400, "INVALID_PARAMETER", f"type debe ser uno de {sorted(TYPES)}")


def check_sensor(s: str) -> None:
    if s not in SENSORS:
        raise ApiError(400, "INVALID_PARAMETER", f"sensor debe ser uno de {sorted(SENSORS)}")


def check_date_syntax(d: str) -> None:
    if len(d) != 10 or d[4] != "-" or d[7] != "-":
        raise ApiError(400, "INVALID_PARAMETER", f"Formato de fecha inválido: {d} (usar YYYY-MM-DD)")
    try:
        dt_date.fromisoformat(d)
    except ValueError:
        raise ApiError(400, "INVALID_PARAMETER", f"Formato de fecha inválido: {d} (usar YYYY-MM-DD)")


def check_period_syntax(p: str) -> None:
    if len(p) != 7 or p[4] != "-":
        raise ApiError(400, "INVALID_PARAMETER", f"Formato de período inválido: {p} (usar YYYY-MM)")
    try:
        dt_date.fromisoformat(f"{p}-01")
    except ValueError:
        raise ApiError(400, "INVALID_PARAMETER", f"Formato de período inválido: {p} (usar YYYY-MM)")


def _date_in_ranges(d: str, ranges: object) -> bool:
    if isinstance(ranges, dict):
        ranges = [ranges]
    if not isinstance(ranges, list):
        return False
    for r in ranges:
        if not isinstance(r, dict):
            continue
        start = r.get("start") or r.get("from")
        end = r.get("end") or r.get("to")
        if start and end and start <= d <= end:
            return True
    return False


def _period_in_ranges(p: str, ranges: object) -> bool:
    if isinstance(ranges, dict):
        ranges = [ranges]
    if not isinstance(ranges, list):
        return False
    for r in ranges:
        if not isinstance(r, dict):
            continue
        start = r.get("start") or r.get("from")
        end = r.get("end") or r.get("to")
        if start and end and start[:7] <= p <= end[:7]:
            return True
    return False


def check_date(
    d: str | None = None,
    sensor: str | None = None,
    full: bool = False,
    comparison: bool = False,
    env_only: bool = False,
    period: str | None = None,
) -> str:
    if d is not None:
        check_date_syntax(d)
        resolved_period = d[:7]
        ok = True
        if sensor:
            ranges_map = META.get("available_date_ranges_by_sensor", {})
            if sensor in ranges_map:
                ok &= _date_in_ranges(d, ranges_map[sensor])
            else:
                ok &= resolved_period in META.get("available_periods_by_sensor", {}).get(sensor, [])
        if full or env_only:
            if "common_environmental_dates" in META:
                ok &= _date_in_ranges(d, META.get("common_environmental_dates"))
            else:
                ok &= resolved_period in (META.get("common_environmental_periods") or [])
        if comparison:
            if "common_analysis_dates" in META:
                ok &= _date_in_ranges(d, META.get("common_analysis_dates"))
            else:
                ok &= resolved_period in (META.get("common_analysis_periods") or [])
        if not ok:
            raise ApiError(422, "PERIOD_NOT_AVAILABLE", f"La fecha {d} no está publicada para esta consulta")
        return resolved_period

    if period is not None:
        check_period_syntax(period)
        ok = True
        if sensor:
            periods_map = META.get("available_periods_by_sensor", {})
            if sensor in periods_map:
                ok &= period in periods_map[sensor]
            else:
                ok &= _period_in_ranges(period, META.get("available_date_ranges_by_sensor", {}).get(sensor))
        if full or env_only:
            if "common_environmental_periods" in META:
                ok &= period in (META.get("common_environmental_periods") or [])
            else:
                ok &= _period_in_ranges(period, META.get("common_environmental_dates"))
        if comparison:
            if "common_analysis_periods" in META:
                ok &= period in (META.get("common_analysis_periods") or [])
            else:
                ok &= _period_in_ranges(period, META.get("common_analysis_dates"))
        if not ok:
            raise ApiError(422, "PERIOD_NOT_AVAILABLE", f"El período {period} no está publicado para esta consulta")
        return period

    raise ApiError(400, "INVALID_PARAMETER", "Debe indicar date (YYYY-MM-DD) o period (YYYY-MM)")


def _aggregate_env_metric_for_period(
    t_type: str,
    period: str,
    metric: str,
    sensor: str | None = None,
) -> list[dict]:
    """Agrega una variable ambiental diaria desde DAILY para un período YYYY-MM."""
    by_terr_day: dict[str, dict[str, dict]] = {}
    for row in DAILY:
        if row.get("territory_type") != t_type:
            continue
        row_date = row.get("date", "")
        if not row_date.startswith(period):
            continue
        if sensor and row.get("sensor") != sensor:
            continue
        tid = str(row.get("territory_id"))
        day_map = by_terr_day.setdefault(tid, {})
        existing = day_map.get(row_date)
        if existing is None or (existing.get(metric) is None and row.get(metric) is not None):
            day_map[row_date] = row

    items: list[dict] = []
    for day_map in by_terr_day.values():
        rows = list(day_map.values())
        val: float | None = None
        if metric == "wind_dir_deg":
            u_acc = 0.0
            v_acc = 0.0
            n_valid = 0
            for r in rows:
                wd = r.get("wind_dir_deg")
                if wd is None:
                    continue
                ws = r.get("wind_speed_10m_ms")
                w = float(ws) if ws is not None else 1.0
                rad = math.radians(float(wd))
                u_acc += -w * math.sin(rad)
                v_acc += -w * math.cos(rad)
                n_valid += 1
            if n_valid > 0 and (abs(u_acc) > 1e-9 or abs(v_acc) > 1e-9):
                val = round((math.degrees(math.atan2(-u_acc, -v_acc)) + 360.0) % 360.0, 1)
        else:
            vals = [float(r[metric]) for r in rows if r.get(metric) is not None]
            if vals:
                val = round(sum(vals) / len(vals), 3)
        items.append({"territory_type": t_type, "territory_id": rows[0].get("territory_id"), "value": val})
    return items


def _build_summary_data_quality(summary_row: dict, assess_row: dict) -> dict:
    """Construye el bloque data_quality combinando cobertura satelital y evaluación ambiental."""
    valid_days = summary_row.get("valid_land_pixel_days")
    expected_days = summary_row.get("expected_land_pixel_days")
    obs_cov = summary_row.get("observation_coverage")
    assess_status = assess_row.get("data_quality_status")

    notes: list[str] = []
    if obs_cov is None or valid_days is None or valid_days <= 0 or obs_cov < 0.4 or assess_status == "insufficient":
        status = "insufficient"
        if obs_cov is None or (valid_days is not None and valid_days <= 0) or (obs_cov is not None and obs_cov < 0.4):
            notes.append("Cobertura de observación satelital insuficiente en el período.")
    elif obs_cov < 0.7 or assess_status == "partial":
        status = "partial"
        if obs_cov < 0.7:
            notes.append("Cobertura de observación satelital parcial (<70 %).")
    else:
        status = "usable"

    missing_vars = assess_row.get("missing_variables") or []
    if missing_vars:
        notes.append(f"Variables ambientales faltantes: {', '.join(missing_vars)}.")

    return {
        "valid_land_pixel_days": valid_days,
        "expected_land_pixel_days": expected_days,
        "observation_coverage": obs_cov,
        "status": status,
        "notes": notes,
    }


def _build_summary_provenance(
    t_type: str,
    t_id: str,
    sensor: str,
    resolved_period: str,
    daily_row: dict | None,
    summary_row: dict,
    assess_row: dict,
) -> dict:
    """Construye el bloque provenance con trazabilidad de sensor, DEM y versión."""
    source_product = summary_row.get("source_product") or (daily_row or {}).get("source_product")
    assess_products = assess_row.get("source_products") or []
    source_products: list[str] = []
    for prod in ([source_product] if source_product else []) + list(assess_products):
        if prod and prod not in source_products:
            source_products.append(prod)

    dem_source = (daily_row or {}).get("dem_source")
    if not dem_source:
        for r in DAILY:
            if (
                r.get("territory_type") == t_type
                and str(r.get("territory_id")) == t_id
                and r.get("sensor") == sensor
                and r.get("date", "").startswith(resolved_period)
                and r.get("dem_source")
            ):
                dem_source = r.get("dem_source")
                break

    processing_version = (
        summary_row.get("processing_version")
        or (daily_row or {}).get("processing_version")
        or assess_row.get("processing_version")
        or META.get("processing_version")
    )
    return {
        "source_product": source_product,
        "source_products": source_products,
        "processing_version": processing_version,
        "data_origin": META.get("data_origin"),
        "dem_source": dem_source,
        "generated_at": META.get("generated_at"),
    }


def find_territory(t_type: str, t_id: str) -> dict:
    for feat in TERR.get("features", []):
        props = feat.get("properties", {})
        if props.get("territory_type") == t_type and str(props.get("territory_id")) == t_id:
            return props
    raise ApiError(404, "NOT_FOUND", f"Territorio {t_type}/{t_id} no encontrado")


def pick(items: list[dict], **keys) -> dict | None:
    return next((i for i in items if all(str(i.get(k)) == str(v) for k, v in keys.items())), None)


@app.get("/")
def root_probe():
    """Sonda liviana en '/' para warm-up / health checks por defecto en Azure App Service."""
    return {
        "service": "McHanaq API",
        "status": "ok",
        "health": "/api/v1/health",
    }


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "data_loaded": bool(META and TERR.get("features") and DAILY and SUMMARY),
        "data_dir": str(DATA),
    }


@app.get("/api/v1/metadata")
def metadata():
    return {"schema_version": "1.0", **META}


@app.get("/api/v1/territories")
def territories(type: str = Query(...)):
    check_type(type)
    feats = [f for f in TERR.get("features", []) if f.get("properties", {}).get("territory_type") == type]
    return {"schema_version": "1.0", "type": "FeatureCollection", "features": feats}


@app.get("/api/v1/territory-metrics")
def territory_metrics(
    type: str = Query(...),
    metric: str = Query(...),
    date: str | None = Query(default=None),
    period: str | None = Query(default=None),
    sensor: str | None = Query(default=None),
):
    check_type(type)
    if metric not in ACTIVITY_METRICS and metric not in ENV_METRICS:
        raise ApiError(400, "INVALID_PARAMETER", f"metric '{metric}' no permitida")
    if metric in ACTIVITY_METRICS and not sensor:
        raise ApiError(400, "INVALID_PARAMETER", f"sensor es obligatorio para la métrica {metric}")
    if sensor:
        check_sensor(sensor)

    resolved_period = check_date(
        d=date,
        period=None if date else period,
        sensor=sensor,
        env_only=(metric in ENV_METRICS and not sensor),
    )

    items = []
    if metric == "conditions_classification":
        for row in ASSESS:
            if row.get("territory_type") == type and row.get("period_id") == resolved_period:
                items.append({
                    "territory_type": type,
                    "territory_id": row.get("territory_id"),
                    "value": row.get("classification"),
                })
    elif metric in MONTHLY_SUMMARY_METRICS or (not date and metric in ACTIVITY_METRICS):
        seen_ids: set[str] = set()
        for row in SUMMARY:
            tid = str(row.get("territory_id"))
            if row.get("territory_type") == type and row.get("period_id") == resolved_period and (not sensor or row.get("sensor") == sensor):
                if not sensor and tid in seen_ids:
                    continue
                seen_ids.add(tid)
                items.append({"territory_type": type, "territory_id": row.get("territory_id"), "value": row.get(metric)})
    elif date:
        by_tid: dict[str, dict] = {}
        for row in DAILY:
            if row.get("territory_type") == type and row.get("date") == date and (not sensor or row.get("sensor") == sensor):
                tid = str(row.get("territory_id"))
                prev = by_tid.get(tid)
                if prev is None or (prev.get(metric) is None and row.get(metric) is not None):
                    by_tid[tid] = row
        items = [
            {"territory_type": type, "territory_id": r.get("territory_id"), "value": r.get(metric)}
            for r in by_tid.values()
        ]
    else:
        items = _aggregate_env_metric_for_period(type, resolved_period, metric, sensor=sensor)

    return {
        "schema_version": "1.0",
        "date": date,
        "period": resolved_period,
        "metric": metric,
        "unit": META.get("units", {}).get(metric),
        "sensor": sensor,
        "items": items,
    }


@app.get("/api/v1/territories/{type}/{id}/summary")
def territory_summary(
    type: str,
    id: str,
    sensor: str = Query(...),
    date: str | None = Query(default=None),
    period: str | None = Query(default=None),
):
    check_type(type)
    check_sensor(sensor)
    resolved_period = check_date(d=date, period=None if date else period, sensor=sensor, full=True)
    terr = find_territory(type, id)

    daily_row = pick(DAILY, territory_type=type, territory_id=id, sensor=sensor, date=date) if date else None
    summary_row = pick(SUMMARY, territory_type=type, territory_id=id, sensor=sensor, period_id=resolved_period) or {}
    assess_row = pick(ASSESS, territory_type=type, territory_id=id, period_id=resolved_period) or {}

    return {
        "schema_version": "1.0",
        "territory": {"type": type, "id": id, "name": terr.get("name") or terr.get("territory_name")},
        "date": date,
        "period": {"id": resolved_period},
        "sensor": sensor,
        "daily": daily_row,
        "daily_context": daily_row,
        "activity": {
            "fire_pixel_days_8_9": summary_row.get("fire_pixel_days_8_9"),
            "active_days": summary_row.get("active_days"),
            "max_consecutive_active_days": summary_row.get("max_consecutive_active_days"),
            "activity_rate": summary_row.get("activity_rate"),
            "observation_coverage": summary_row.get("observation_coverage"),
            "historical_anomaly": summary_row.get("historical_anomaly", {"value": None, "status": "insufficient_baseline", "baseline_years": 0}),
        },
        "environmental_conditions": {
            "classification": assess_row.get("classification", "not_evaluable"),
            "decision_path": assess_row.get("decision_path", []),
            "factors": assess_row.get("evidence", []),
            "evidence": assess_row.get("evidence", []),
            "modifiers": assess_row.get("modifiers", []),
            "missing_variables": assess_row.get("missing_variables", []),
        },
        "data_quality": _build_summary_data_quality(summary_row, assess_row),
        "provenance": _build_summary_provenance(type, id, sensor, resolved_period, daily_row, summary_row, assess_row),
    }


@app.get("/api/v1/territories/{type}/{id}/activity")
def territory_activity(
    type: str,
    id: str,
    sensor: str = Query(...),
    from_date: str | None = Query(default=None, alias="from"),
    to_date: str | None = Query(default=None, alias="to"),
    period: str | None = Query(default=None),
):
    check_type(type)
    check_sensor(sensor)
    if from_date and to_date:
        check_date_syntax(from_date)
        check_date_syntax(to_date)
        if from_date > to_date:
            raise ApiError(400, "INVALID_PARAMETER", "from debe ser menor o igual a to")
        check_date(from_date, sensor=sensor)
        check_date(to_date, sensor=sensor)
    elif from_date or to_date:
        raise ApiError(400, "INVALID_PARAMETER", "Debe indicar ambos parámetros from y to (YYYY-MM-DD)")
    elif period:
        check_date(period=period, sensor=sensor)
    else:
        raise ApiError(400, "INVALID_PARAMETER", "Debe indicar from y to (YYYY-MM-DD) o period (YYYY-MM)")

    find_territory(type, id)
    rows = [
        r for r in DAILY
        if r.get("territory_type") == type
        and str(r.get("territory_id")) == id
        and r.get("sensor") == sensor
        and ((from_date <= r.get("date", "") <= to_date) if (from_date and to_date) else r.get("date", "").startswith(period))
    ]
    rows.sort(key=lambda r: r.get("date", ""))
    return {
        "schema_version": "1.0",
        "territory": {"type": type, "id": id},
        "territory_type": type,
        "territory_id": id,
        "sensor": sensor,
        "from": from_date,
        "to": to_date,
        "period": None if (from_date and to_date) else period,
        "items": rows,
    }


@app.get("/api/v1/territories/{type}/{id}/sensor-comparison")
def sensor_comparison(
    type: str,
    id: str,
    date: str | None = Query(default=None),
    period: str | None = Query(default=None),
):
    check_type(type)
    resolved_period = check_date(d=date, period=None if date else period, comparison=True)
    find_territory(type, id)

    row = pick(COMPARE, territory_type=type, territory_id=id, period_id=resolved_period)
    if not row:
        return {
            "schema_version": "1.0",
            "date": date,
            "territory_type": type,
            "territory_id": id,
            "period_id": resolved_period,
            "status": "insufficient",
            "sensors": {},
            "interpretation": "No hay datos de comparación multisensor disponibles para este territorio y período.",
            "method_version": META.get("processing_version"),
        }
    return {"schema_version": "1.0", "date": date, **row}


@app.get("/api/v1/territories/{type}/{id}/export.csv")
def export_csv(
    type: str,
    id: str,
    sensor: str = Query(...),
    from_date: str = Query(..., alias="from"),
    to_date: str = Query(..., alias="to"),
):
    check_type(type)
    check_sensor(sensor)
    check_date_syntax(from_date)
    check_date_syntax(to_date)
    if from_date > to_date:
        raise ApiError(400, "INVALID_PARAMETER", "from debe ser menor o igual a to")
    check_date(from_date, sensor=sensor)
    check_date(to_date, sensor=sensor)
    find_territory(type, id)

    rows = [
        r for r in DAILY
        if r.get("territory_type") == type
        and str(r.get("territory_id")) == id
        and r.get("sensor") == sensor
        and from_date <= r.get("date", "") <= to_date
    ]
    rows.sort(key=lambda r: r.get("date", ""))
    buf = io.StringIO()
    cols = [
        "date", "territory_type", "territory_id", "sensor",
        "fire_pixels_low", "fire_pixels_nominal_high", "valid_land_pixels",
        "expected_land_pixels_per_day", "observation_coverage", "activity_rate",
        "t_amb_c", "relative_humidity_2m_pct", "t_suelo_c", "lst_valid_pixel_pct",
        "wind_dir_deg", "wind_speed_10m_ms", "elevation_m", "dem_source",
        "source_product", "processing_version",
    ]
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    writer.writerows({c: ("" if r.get(c) is None else r.get(c)) for c in cols} for r in rows)
    buf.seek(0)
    filename = f"mchanaq_{type}_{id}_{sensor}_{from_date}_{to_date}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/v1/territories/{type}/{id}/calendar")
def territory_calendar(
    type: str,
    id: str,
    sensor: str = Query(...),
    from_period: str | None = Query(default=None),
    to_period: str | None = Query(default=None),
):
    check_type(type)
    check_sensor(sensor)
    if from_period is not None:
        check_period_syntax(from_period)
    if to_period is not None:
        check_period_syntax(to_period)
    if from_period and to_period and from_period > to_period:
        raise ApiError(400, "INVALID_PARAMETER", "from_period debe ser menor o igual a to_period")
    find_territory(type, id)

    rows = [
        r for r in SUMMARY
        if r.get("territory_type") == type
        and str(r.get("territory_id")) == id
        and r.get("sensor") == sensor
        and (not from_period or r.get("period_id", "") >= from_period)
        and (not to_period or r.get("period_id", "") <= to_period)
    ]
    rows.sort(key=lambda r: r.get("period_id", ""))
    items = [
        {
            "period_id": r.get("period_id"),
            "fire_pixel_days_8_9": r.get("fire_pixel_days_8_9"),
            "fire_pixel_days_7_9": r.get("fire_pixel_days_7_9"),
            "activity_rate": r.get("activity_rate"),
            "observation_coverage": r.get("observation_coverage"),
            "active_days": r.get("active_days"),
            "max_consecutive_active_days": r.get("max_consecutive_active_days"),
            "historical_anomaly": r.get(
                "historical_anomaly",
                {"value": None, "status": "insufficient_baseline", "baseline_years": 0},
            ),
            "source_product": r.get("source_product"),
            "processing_version": r.get("processing_version"),
        }
        for r in rows
    ]
    return {
        "schema_version": "1.0",
        "territory": {"type": type, "id": id},
        "sensor": sensor,
        "from_period": from_period,
        "to_period": to_period,
        "items": items,
    }


@app.get("/api/v1/territories/{type}/{id}/hotspots")
def territory_hotspots(type: str, id: str, date: str = Query(...), sensor: str = Query(...)):
    check_type(type)
    check_sensor(sensor)
    check_date(date, sensor=sensor)
    find_territory(type, id)
    if not HOTSPOTS:
        raise ApiError(404, "NOT_FOUND", "hotspots.geojson no está disponible")
    feats = [
        f for f in HOTSPOTS.get("features", [])
        if f.get("properties", {}).get("territory_type") == type
        and str(f.get("properties", {}).get("territory_id")) == id
        and f.get("properties", {}).get("date") == date
        and f.get("properties", {}).get("sensor") == sensor
    ]
    return {"schema_version": "1.0", "type": "FeatureCollection", "features": feats}


# ---------------------------------------------------------------------------
# Fase 2 (Opcional y desacoplada) — Simulación espacial de propagación (48 h)
# ---------------------------------------------------------------------------

SPREAD_FORECASTS = load_json("spread_forecasts.json") or {"items": []}
SPREAD_OUTPUTS_DIR = (DATA / "spread_outputs") if (DATA / "spread_outputs").exists() else (ROOT / "data" / "processed" / "spread_outputs")
_SPREAD_MIME_TYPES = {
    "simulacion_48h.gif": "image/gif",
    "pronostico_48h.mp4": "video/mp4",
    "mchanaq_propagacion_48h_qgis.zip": "application/zip",
    "infografia_impacto_48h.png": "image/png",
    "panel_modelo_48h.png": "image/png",
}


def _territory_centroid_and_geom(terr: dict):
    from shapely.geometry import shape

    geom = shape(terr["geometry"]) if terr.get("geometry") else None
    if geom is not None and not geom.is_empty:
        pt = geom.representative_point()
        return float(pt.y), float(pt.x), geom
    return -8.19833, -75.26669, None


def _fetch_open_meteo_dem_slope_py(lat: float, lon: float) -> tuple[float | None, float | None, str]:
    """Calcula pendiente media/máxima con Open-Meteo Elevation (Copernicus GLO-90) usando solo httpx."""
    half = 0.032
    pts = [
        (round(lat - half + (2 * half * r / 3.0), 5), round(lon - half + (2 * half * c / 3.0), 5))
        for r in range(4)
        for c in range(4)
    ]
    try:
        with httpx.Client(timeout=8.0) as client:
            resp = client.get(
                "https://api.open-meteo.com/v1/elevation",
                params={
                    "latitude": ",".join(str(p[0]) for p in pts),
                    "longitude": ",".join(str(p[1]) for p in pts),
                },
            )
            resp.raise_for_status()
            elev = resp.json().get("elevation") or []
        if len(elev) < 16:
            return None, None, "COPERNICUS_DEM_GLO30_PARAMETRIC"
        grid = [[float(elev[r * 4 + c]) for c in range(4)] for r in range(4)]
        dx = (2 * half / 3.0) * 111_320.0 * max(0.2, math.cos(math.radians(lat)))
        dy = (2 * half / 3.0) * 110_540.0
        slopes: list[float] = []
        for r in range(3):
            for c in range(3):
                dz_x = (grid[r][c + 1] - grid[r][c]) / max(dx, 1.0)
                dz_y = (grid[r + 1][c] - grid[r][c]) / max(dy, 1.0)
                slopes.append(math.degrees(math.atan(math.hypot(dz_x, dz_y))))
        mean_deg = round(sum(slopes) / len(slopes), 2)
        max_deg = round(max(slopes), 2)
        return mean_deg, max_deg, "COPERNICUS_DEM_GLO90_OPEN_METEO"
    except Exception:
        return None, None, "COPERNICUS_DEM_GLO30_PARAMETRIC"


def _run_lightweight_global_spread(
    *,
    t_type: str,
    t_id: str,
    t_name: str,
    lat: float,
    lon: float,
    date_str: str,
    horizon_h: int,
    ws: float,
    wd: float,
    veg: float,
    slp: float,
    fm: float,
    use_open_meteo: bool,
) -> dict:
    """Fallback sin geopandas/numpy: obtiene Copernicus GLO-90 DEM + viento Open-Meteo reales vía httpx."""
    slope_mean, slope_max, dem_source = _fetch_open_meteo_dem_slope_py(lat, lon)
    eff_slope = slope_mean if slope_mean is not None else round(float(slp), 2)
    eff_max_slope = slope_max if slope_max is not None else round(eff_slope * 1.8, 2)

    wind_source = "era5_daily_diurnal_profile"
    eff_ws, eff_wd = float(ws), float(wd)
    if use_open_meteo:
        live = _fetch_open_meteo_current(lat, lon)
        if live.get("status") == "ok":
            wind_source = "Open-Meteo"
            if isinstance(live.get("wind_speed_10m_ms"), (int, float)):
                eff_ws = float(live["wind_speed_10m_ms"])
            if isinstance(live.get("wind_dir_deg"), (int, float)):
                eff_wd = float(live["wind_dir_deg"])

    moisture_factor = max(0.0, 1.0 - min(1.0, fm / 30.0)) ** 2
    slope_factor = min(8.0, math.exp(0.0693 * max(0.0, eff_slope)))
    head_ros = round(1.8 * (1.0 + 0.72 * (max(0.0, eff_ws) ** 1.22)) * moisture_factor * (0.4 + 0.6 * (veg / 100.0)) * slope_factor, 2)
    lb = round(min(8.0, max(1.0, 0.936 * math.exp(0.2566 * eff_ws) + 0.461 * math.exp(-0.1548 * eff_ws) - 0.397)), 2)

    return {
        "schema_version": "1.0",
        "phase": "phase_2_optional",
        "territory": {"type": t_type, "id": t_id, "name": t_name},
        "date": date_str,
        "horizon_h": horizon_h,
        "ignition": {"lat": lat, "lon": lon},
        "dominant_wind": {"wind_dir_deg": round(eff_wd, 1), "wind_speed_10m_ms": round(eff_ws, 2)},
        "savi_summary": {"mean": round(0.12 + 0.66 * (veg / 100.0), 3), "source": "vegetation_area_pct"},
        "slope_summary": {"mean_deg": eff_slope, "max_deg": eff_max_slope, "dem_source": dem_source},
        "spread_summary": {"head_ros_m_min": head_ros, "length_to_breadth": lb},
        "data_sources": {
            "wind_source": wind_source,
            "dem_source": dem_source,
            "vegetation_index": "vegetation_area_pct",
        },
        "timings_s": {"simulation_s": 0.05, "total_s": 0.05},
    }


def _run_phase2_simulation(
    t_type: str,
    t_id: str,
    t_name: str,
    geom,
    lat: float,
    lon: float,
    date_str: str,
    horizon_h: int,
    wind_speed_ms: float | None = None,
    wind_dir_deg: float | None = None,
    vegetation_pct: float | None = None,
    slope_mean_deg: float | None = None,
    fuel_moisture_pct: float | None = None,
    use_open_meteo: bool = False,
    export_products: bool = False,
) -> dict:
    daily_match = next(
        (
            r for r in DAILY
            if r.get("territory_type") == t_type
            and str(r.get("territory_id")) == t_id
            and r.get("date") == date_str
        ),
        None,
    )
    assess_match = next(
        (
            r for r in ASSESS
            if r.get("territory_type") == t_type
            and str(r.get("territory_id")) == t_id
            and r.get("period_id") == date_str[:7]
        ),
        None,
    )

    def _ev_num(var_name: str) -> float | None:
        if not assess_match:
            return None
        for ev in assess_match.get("evidence") or []:
            if ev.get("variable") == var_name and isinstance(ev.get("value"), (int, float)):
                return float(ev["value"])
        return None

    ws = wind_speed_ms if wind_speed_ms is not None else (
        float(daily_match["wind_speed_10m_ms"])
        if daily_match and daily_match.get("wind_speed_10m_ms") is not None
        else (_ev_num("wind_speed_10m_ms") or 4.6)
    )
    wd = wind_dir_deg if wind_dir_deg is not None else (
        float(daily_match["wind_dir_deg"])
        if daily_match and daily_match.get("wind_dir_deg") is not None
        else 118.0
    )
    veg = vegetation_pct if vegetation_pct is not None else (_ev_num("vegetation_area_pct") or 82.0)
    slope_pct = _ev_num("slope_class_area_pct")
    slp = slope_mean_deg if slope_mean_deg is not None else (
        round(math.degrees(math.atan(slope_pct / 100.0)), 1) if slope_pct is not None else 6.5
    )
    precip = _ev_num("precip_30d_mm")
    fm = fuel_moisture_pct if fuel_moisture_pct is not None else (
        max(4.0, min(28.0, round(4.0 + precip * 0.18, 1))) if precip is not None else 9.0
    )

    t0 = time.perf_counter()
    try:
        from src.analytics.spread_model import (
            build_spread_forecast_payload,
            export_all_phase2_products,
            simulate_spread_48h,
        )
    except ImportError as exc:
        if export_products:
            raise ApiError(
                503,
                "EXPORT_LIBS_UNAVAILABLE",
                f"Generación GIS/video requiere geopandas/matplotlib en este entorno ({exc}). Las descargas precalculadas siguen activas.",
            )
        return _run_lightweight_global_spread(
            t_type=t_type,
            t_id=t_id,
            t_name=t_name,
            lat=lat,
            lon=lon,
            date_str=date_str,
            horizon_h=horizon_h,
            ws=ws,
            wd=wd,
            veg=veg,
            slp=slp,
            fm=fm,
            use_open_meteo=use_open_meteo,
        )

    res = simulate_spread_48h(
        territory_type=t_type,
        territory_id=t_id,
        territory_name=t_name,
        territory_geom=geom,
        date_str=date_str,
        ignition_lat=lat,
        ignition_lon=lon,
        horizon_h=horizon_h,
        vegetation_pct=veg,
        slope_mean_deg=slp,
        fuel_moisture_pct=fm,
        base_wind_ms=ws,
        base_wind_dir_deg=wd,
        use_open_meteo=use_open_meteo,
    )
    sim_s = round(time.perf_counter() - t0, 3)
    timings: dict[str, float] = {"simulation_s": sim_s, "total_s": sim_s}
    if export_products:
        exported = export_all_phase2_products(res, SPREAD_OUTPUTS_DIR)
        timings.update(exported["timings_s"])
        timings["total_s"] = round(sim_s + exported["timings_s"]["total_export_s"], 3)
    return build_spread_forecast_payload(res, timings_s=timings)


def _realtime_enabled() -> bool:
    """AGENTS.md: el backend oficial no consulta almacenamiento/APIs externas por
    petición HTTP salvo la excepción explícita de Fase 2. Estos endpoints de
    tiempo real quedan apagados por defecto y solo se activan a propósito con
    MCHANAQ_ENABLE_REALTIME=1 (por ejemplo, para una demo en vivo)."""
    return os.getenv("MCHANAQ_ENABLE_REALTIME", "").strip().lower() in {"1", "true", "yes"}


def _dispatch_spread_request(
    *,
    lat: float | None,
    lon: float | None,
    date: str,
    horizon_h: int,
    wind_speed_10m_ms: float | None,
    wind_dir_deg: float | None,
    vegetation_pct: float | None,
    slope_mean_deg: float | None,
    fuel_moisture_pct: float | None,
    use_open_meteo: bool,
    payload: dict | None,
    export_products: bool,
) -> dict:
    body = payload or {}
    r_lat = float(body["lat"]) if "lat" in body else lat
    r_lon = float(body["lon"]) if "lon" in body else lon
    if r_lat is None or r_lon is None:
        raise ApiError(400, "INVALID_PARAMETER", "Debe indicar lat y lon para simular propagación")
    if not (-90.0 <= r_lat <= 90.0 and -180.0 <= r_lon <= 180.0):
        raise ApiError(400, "INVALID_PARAMETER", "Coordenadas fuera de rango global")

    r_date = str(body.get("date") or date)
    check_date_syntax(r_date)
    r_horizon = int(body.get("horizon_h") or horizon_h)
    if not (6 <= r_horizon <= 48):
        raise ApiError(400, "INVALID_PARAMETER", "horizon_h debe estar entre 6 y 48 horas")

    r_open_meteo = bool(body["use_open_meteo"]) if "use_open_meteo" in body else use_open_meteo
    if r_open_meteo and not _realtime_enabled():
        raise ApiError(
            403,
            "REALTIME_DISABLED",
            "use_open_meteo requiere MCHANAQ_ENABLE_REALTIME=1 (desactivado por defecto; ver AGENTS.md).",
        )
    return _run_phase2_simulation(
        t_type=str(body.get("territory_type") or "custom"),
        t_id=str(body.get("territory_id") or "custom"),
        t_name=str(body.get("territory_name") or f"Punto ({r_lat:.3f}, {r_lon:.3f})"),
        geom=None,
        lat=r_lat,
        lon=r_lon,
        date_str=r_date,
        horizon_h=r_horizon,
        wind_speed_ms=float(body["wind_speed_10m_ms"]) if "wind_speed_10m_ms" in body else wind_speed_10m_ms,
        wind_dir_deg=float(body["wind_dir_deg"]) if "wind_dir_deg" in body else wind_dir_deg,
        vegetation_pct=float(body["vegetation_pct"]) if "vegetation_pct" in body else vegetation_pct,
        slope_mean_deg=float(body["slope_mean_deg"]) if "slope_mean_deg" in body else slope_mean_deg,
        fuel_moisture_pct=float(body["fuel_moisture_pct"]) if "fuel_moisture_pct" in body else fuel_moisture_pct,
        use_open_meteo=r_open_meteo,
        export_products=export_products,
    )


@app.get("/api/v1/territories/{type}/{id}/spread-forecast")
def territory_spread_forecast(
    type: str,
    id: str,
    date: str | None = Query(default=None),
    horizon_h: int = Query(default=48, ge=6, le=48),
):
    check_type(type)
    terr = find_territory(type, id)
    if date is not None:
        check_date_syntax(date)

    for item in SPREAD_FORECASTS.get("items", []):
        t_info = item.get("territory") or {}
        if (
            t_info.get("type") == type
            and str(t_info.get("id")) == id
            and (date is None or item.get("date") == date)
            and int(item.get("horizon_h") or 48) == horizon_h
        ):
            return item

    lat, lon, geom = _territory_centroid_and_geom(terr)
    t_name = str(terr.get("name") or terr.get("territory_name") or id)
    resolved_date = date or (
        SPREAD_FORECASTS.get("items", [{}])[0].get("date")
        if SPREAD_FORECASTS.get("items")
        else "2024-09-25"
    )
    return _run_phase2_simulation(
        t_type=type,
        t_id=id,
        t_name=t_name,
        geom=geom,
        lat=lat,
        lon=lon,
        date_str=resolved_date,
        horizon_h=horizon_h,
    )


@app.post("/api/v1/spread-simulate")
@app.get("/api/v1/spread-simulate")
def spread_simulate(
    lat: float | None = Query(default=None, ge=-90.0, le=90.0),
    lon: float | None = Query(default=None, ge=-180.0, le=180.0),
    date: str = Query(default="2024-09-25"),
    horizon_h: int = Query(default=48, ge=6, le=48),
    wind_speed_10m_ms: float | None = Query(default=None, ge=0.0, le=40.0),
    wind_dir_deg: float | None = Query(default=None, ge=0.0, le=360.0),
    vegetation_pct: float | None = Query(default=None, ge=0.0, le=100.0),
    slope_mean_deg: float | None = Query(default=None, ge=0.0, le=60.0),
    fuel_moisture_pct: float | None = Query(default=None, ge=1.0, le=35.0),
    use_open_meteo: bool = Query(default=False),
    payload: dict | None = Body(default=None),
):
    return _dispatch_spread_request(
        lat=lat,
        lon=lon,
        date=date,
        horizon_h=horizon_h,
        wind_speed_10m_ms=wind_speed_10m_ms,
        wind_dir_deg=wind_dir_deg,
        vegetation_pct=vegetation_pct,
        slope_mean_deg=slope_mean_deg,
        fuel_moisture_pct=fuel_moisture_pct,
        use_open_meteo=use_open_meteo,
        payload=payload,
        export_products=False,
    )


@app.post("/api/v1/spread-export")
@app.get("/api/v1/spread-export")
def spread_export(
    lat: float | None = Query(default=None, ge=-90.0, le=90.0),
    lon: float | None = Query(default=None, ge=-180.0, le=180.0),
    date: str = Query(default="2024-09-25"),
    horizon_h: int = Query(default=48, ge=6, le=48),
    wind_speed_10m_ms: float | None = Query(default=None, ge=0.0, le=40.0),
    wind_dir_deg: float | None = Query(default=None, ge=0.0, le=360.0),
    vegetation_pct: float | None = Query(default=None, ge=0.0, le=100.0),
    slope_mean_deg: float | None = Query(default=None, ge=0.0, le=60.0),
    fuel_moisture_pct: float | None = Query(default=None, ge=1.0, le=35.0),
    use_open_meteo: bool = Query(default=False),
    payload: dict | None = Body(default=None),
):
    return _dispatch_spread_request(
        lat=lat,
        lon=lon,
        date=date,
        horizon_h=horizon_h,
        wind_speed_10m_ms=wind_speed_10m_ms,
        wind_dir_deg=wind_dir_deg,
        vegetation_pct=vegetation_pct,
        slope_mean_deg=slope_mean_deg,
        fuel_moisture_pct=fuel_moisture_pct,
        use_open_meteo=use_open_meteo,
        payload=payload,
        export_products=True,
    )


@app.get("/api/v1/spread-outputs/{filename}")
def spread_output_file(filename: str):
    if filename not in _SPREAD_MIME_TYPES:
        raise ApiError(404, "NOT_FOUND", f"Archivo de salida de Fase 2 no encontrado: {filename}")
    target = SPREAD_OUTPUTS_DIR / filename
    if not target.exists():
        raise ApiError(404, "NOT_FOUND", f"El archivo {filename} aún no fue generado en el servidor")
    return FileResponse(
        path=target,
        media_type=_SPREAD_MIME_TYPES[filename],
        filename=filename,
    )


# ---------------------------------------------------------------------------
# Tiempo real global — Open-Meteo (clima/viento) + NASA FIRMS (MODIS & VIIRS)
# ---------------------------------------------------------------------------

_FIRMS_SOURCES = (
    ("MODIS_NRT", "MODIS"),
    ("VIIRS_SNPP_NRT", "VIIRS_SNPP"),
    ("VIIRS_NOAA20_NRT", "VIIRS_NOAA20"),
)


def _fetch_open_meteo_current(lat: float, lon: float) -> dict:
    """Consulta clima y viento actual en Open-Meteo para el centro del área."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
        cur = payload.get("current") or {}
        return {
            "status": "ok",
            "source": "Open-Meteo",
            "observed_at": cur.get("time"),
            "center": {"lat": round(lat, 4), "lon": round(lon, 4)},
            "t_amb_c": cur.get("temperature_2m"),
            "relative_humidity_2m_pct": cur.get("relative_humidity_2m"),
            "wind_speed_10m_ms": cur.get("wind_speed_10m"),
            "wind_dir_deg": cur.get("wind_direction_10m"),
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "source": "Open-Meteo",
            "detail": str(exc),
        }


def _fetch_firms_source(
    client: httpx.Client,
    map_key: str,
    firms_source: str,
    sensor_label: str,
    west: float,
    south: float,
    east: float,
    north: float,
    days: int,
) -> tuple[list[dict], str | None]:
    """Consulta focos activos NRT en NASA FIRMS Area API sin mezclar sensores."""
    bbox = f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{firms_source}/{bbox}/{days}"
    try:
        resp = client.get(url)
        resp.raise_for_status()
        text = resp.text.strip()
        if not text or text.lower().startswith("invalid"):
            return [], text or "empty_response"
        reader = csv.DictReader(io.StringIO(text))
        features: list[dict] = []
        for row in reader:
            try:
                lat_v = float(row["latitude"])
                lon_v = float(row["longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            frp_raw = row.get("frp")
            try:
                frp_val = float(frp_raw) if frp_raw not in (None, "") else None
            except ValueError:
                frp_val = None
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon_v, lat_v]},
                    "properties": {
                        "sensor": sensor_label,
                        "firms_source": firms_source,
                        "acq_date": row.get("acq_date"),
                        "acq_time": row.get("acq_time"),
                        "confidence": row.get("confidence"),
                        "frp_mw": frp_val,
                        "satellite": row.get("satellite"),
                        "daynight": row.get("daynight"),
                    },
                }
            )
        return features, None
    except Exception as exc:
        return [], str(exc)


_REVERSE_GEOCODE_CACHE: dict[tuple[float, float], dict] = {}
_COUNTRY_SHORT_ALIASES = {
    "Estados Unidos de América": "Estados Unidos",
    "United States of America": "Estados Unidos",
    "United Kingdom": "Reino Unido",
    "Reino Unido de Gran Bretaña e Irlanda del Norte": "Reino Unido",
}


def _dedupe_place_parts(parts: list[str | None]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in parts:
        val = (raw or "").strip()
        if not val:
            continue
        key = val.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def _reverse_geocode_coords(lat: float, lon: float) -> dict:
    """Detecta país, región/estado, provincia/condado y ciudad/distrito vía OSM Nominatim."""
    cache_key = (round(lat, 3), round(lon, 3))
    cached = _REVERSE_GEOCODE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "format": "jsonv2",
        "lat": f"{lat:.5f}",
        "lon": f"{lon:.5f}",
        "zoom": "14",
        "addressdetails": "1",
        "accept-language": "es",
    }
    headers = {"User-Agent": "McHanaq/1.0 (+https://github.com/TomJordan1/McHanaq)"}
    try:
        with httpx.Client(timeout=4.5, headers=headers) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
        addr = payload.get("address") or {}
        if not isinstance(addr, dict) or not addr:
            return {
                "status": "unavailable",
                "source": "OSM_NOMINATIM",
                "country": None,
                "country_code": None,
                "region": None,
                "province": None,
                "city": None,
                "district": None,
                "hierarchy": [],
                "header_primary": None,
                "header_sub": None,
                "display_name": payload.get("display_name"),
            }

        raw_country = (addr.get("country") or "").strip() or None
        country = _COUNTRY_SHORT_ALIASES.get(raw_country or "", raw_country)
        country_code = (addr.get("country_code") or "").upper() or None

        region = (
            addr.get("state")
            or addr.get("region")
            or addr.get("archipelago")
            or addr.get("island")
            or None
        )
        # En Perú y otros países Nominatim coloca la provincia en `region` cuando `state` existe.
        province = addr.get("province") or addr.get("county") or addr.get("state_district")
        if not province and addr.get("state") and addr.get("region") and addr.get("region") != addr.get("state"):
            province = addr.get("region")
        if province and region and province.strip().casefold() == region.strip().casefold():
            province = None

        city = (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("municipality")
            or None
        )
        district = (
            addr.get("city_district")
            or addr.get("suburb")
            or addr.get("borough")
            or addr.get("quarter")
            or addr.get("neighbourhood")
            or addr.get("hamlet")
            or addr.get("locality")
            or None
        )
        if district and city and district.strip().casefold() == city.strip().casefold():
            district = None

        hierarchy = _dedupe_place_parts([country, region, province, city, district])
        sub_parts = _dedupe_place_parts([city, district, province, region])
        # Limita la cabecera superior a máx. 2-3 niveles clave sin redundancia.
        header_sub = " · ".join(sub_parts[:2]) if sub_parts else None
        result = {
            "status": "ok",
            "source": "OSM_NOMINATIM",
            "country": country,
            "country_code": country_code,
            "region": region.strip() if isinstance(region, str) and region.strip() else None,
            "province": province.strip() if isinstance(province, str) and province.strip() else None,
            "city": city.strip() if isinstance(city, str) and city.strip() else None,
            "district": district.strip() if isinstance(district, str) and district.strip() else None,
            "hierarchy": hierarchy,
            "header_primary": country or (sub_parts[0] if sub_parts else None),
            "header_sub": header_sub,
            "display_name": " · ".join(hierarchy) if hierarchy else payload.get("display_name"),
        }
        if len(_REVERSE_GEOCODE_CACHE) >= 512:
            _REVERSE_GEOCODE_CACHE.clear()
        _REVERSE_GEOCODE_CACHE[cache_key] = result
        return dict(result)
    except Exception as exc:
        return {
            "status": "unavailable",
            "source": "OSM_NOMINATIM",
            "country": None,
            "country_code": None,
            "region": None,
            "province": None,
            "city": None,
            "district": None,
            "hierarchy": [],
            "header_primary": None,
            "header_sub": None,
            "display_name": None,
            "detail": str(exc),
        }


@app.get("/api/v1/reverse-geocode")
def reverse_geocode(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
):
    """Resuelve país, región/estado, provincia/condado y ciudad/distrito para cualquier punto del planeta."""
    if not _realtime_enabled():
        return {
            "schema_version": "1.0",
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "status": "realtime_disabled",
            "realtime_setup": {
                "env_var": "MCHANAQ_ENABLE_REALTIME",
                "note": "Configura MCHANAQ_ENABLE_REALTIME=1 en el servidor para habilitar la geocodificación en tiempo real (OSM Nominatim).",
            },
        }
    return {
        "schema_version": "1.0",
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        **_reverse_geocode_coords(lat, lon),
    }


@app.get("/api/v1/realtime/area")
def realtime_area(
    south: float = Query(..., ge=-90.0, le=90.0),
    west: float = Query(..., ge=-180.0, le=180.0),
    north: float = Query(..., ge=-90.0, le=90.0),
    east: float = Query(..., ge=-180.0, le=180.0),
    days: int = Query(3, ge=1, le=10),
):
    """Consulta en tiempo real clima global (Open-Meteo), geocodificación administrativa y focos activos MODIS/VIIRS (NASA FIRMS) para cualquier bbox del mundo."""
    if south >= north or west >= east:
        raise ApiError(400, "INVALID_PARAMETER", "Bounding box inválido: requiere south < north y west < east")

    if not _realtime_enabled():
        return {
            "schema_version": "1.0",
            "bbox": {"south": south, "west": west, "north": north, "east": east},
            "days": days,
            "hotspots_status": "realtime_disabled",
            "counts_by_sensor": {},
            "hotspots": {"type": "FeatureCollection", "features": []},
            "realtime_setup": {
                "env_var": "MCHANAQ_ENABLE_REALTIME",
                "note": "Configura MCHANAQ_ENABLE_REALTIME=1 en el servidor para habilitar clima/geocodificación/focos en tiempo real (Open-Meteo, Nominatim, NASA FIRMS).",
            },
        }

    center_lat = (south + north) / 2.0
    center_lon = (west + east) / 2.0
    weather = _fetch_open_meteo_current(center_lat, center_lon)
    location = _reverse_geocode_coords(center_lat, center_lon)

    firms_key = os.getenv("FIRMS_MAP_KEY", "").strip()
    if not firms_key:
        return {
            "schema_version": "1.0",
            "bbox": {"south": south, "west": west, "north": north, "east": east},
            "days": days,
            "location": location,
            "weather": weather,
            "hotspots_status": "firms_map_key_required",
            "counts_by_sensor": {},
            "hotspots": {"type": "FeatureCollection", "features": []},
            "firms_setup": {
                "env_var": "FIRMS_MAP_KEY",
                "register_url": "https://firms.modaps.eosdis.nasa.gov/api/area/",
                "note": "Configura FIRMS_MAP_KEY en el servidor para habilitar MODIS_NRT, VIIRS_SNPP_NRT y VIIRS_NOAA20_NRT en tiempo real.",
            },
        }

    all_features: list[dict] = []
    counts_by_sensor: dict[str, int] = {}
    errors_by_sensor: dict[str, str] = {}

    with httpx.Client(timeout=15.0) as client:
        for firms_source, sensor_label in _FIRMS_SOURCES:
            feats, err = _fetch_firms_source(
                client=client,
                map_key=firms_key,
                firms_source=firms_source,
                sensor_label=sensor_label,
                west=west,
                south=south,
                east=east,
                north=north,
                days=days,
            )
            counts_by_sensor[sensor_label] = len(feats)
            if err:
                errors_by_sensor[sensor_label] = err
            all_features.extend(feats)

    return {
        "schema_version": "1.0",
        "bbox": {"south": south, "west": west, "north": north, "east": east},
        "days": days,
        "location": location,
        "weather": weather,
        "hotspots_status": "ok" if not errors_by_sensor else "partial",
        "counts_by_sensor": counts_by_sensor,
        "errors_by_sensor": errors_by_sensor or None,
        "hotspots": {"type": "FeatureCollection", "features": all_features},
    }
