#!/usr/bin/env python3

__author__ = "Mengjie"

import argparse, sys, math, json, io, os, re
from typing import Optional, Tuple
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "outputs", ".mplcache"))
import matplotlib
matplotlib.use("Agg")
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from dateutil import tz
import pytz

def _day_of_year(d: date) -> int:
    return (d - date(d.year, 1, 1)).days + 1

def solar_position_series(lat: float, lon: float, local_times: pd.DatetimeIndex):
    lat_rad = math.radians(lat)
    mu = np.zeros(len(local_times))
    m_air = np.zeros(len(local_times))
    sza_deg = np.zeros(len(local_times))
    for i, dt_local in enumerate(local_times):
        tz_offset_min = dt_local.utcoffset().total_seconds() / 60.0
        n = (dt_local.date() - date(dt_local.year, 1, 1)).days + 1
        frac_hour = dt_local.hour + dt_local.minute/60.0 + dt_local.second/3600.0
        gamma = 2.0*math.pi/365.0 * (n - 1 + (frac_hour - 12)/24.0)
        EOT = 229.18*(0.000075 + 0.001868*math.cos(gamma) - 0.032077*math.sin(gamma)
                      - 0.014615*math.cos(2*gamma) - 0.040849*math.sin(2*gamma))
        dec = (0.006918 - 0.399912*math.cos(gamma) + 0.070257*math.sin(gamma)
               - 0.006758*math.cos(2*gamma) + 0.000907*math.sin(2*gamma)
               - 0.002697*math.cos(3*gamma) + 0.00148*math.sin(3*gamma))
        time_offset = EOT + 4.0*lon - tz_offset_min
        tst = (dt_local.hour*60.0 + dt_local.minute + dt_local.second/60.0 + time_offset) % 1440.0
        ha_deg = (tst/4.0) - 180.0
        ha = math.radians(ha_deg)
        cos_zen = math.sin(lat_rad)*math.sin(dec) + math.cos(lat_rad)*math.cos(dec)*math.cos(ha)
        cos_zen = max(cos_zen, 0.0)
        mu[i] = cos_zen
        theta = math.degrees(math.acos(min(1.0, max(-1.0, cos_zen))))
        sza_deg[i] = theta
        if cos_zen > 0:
            m = 1.0 / (cos_zen + 0.50572*((96.07995 - theta)**-1.6364))
        else:
            m = np.nan
        m_air[i] = m
    return sza_deg, mu, m_air

def fetch_open_meteo_uvi(lat: float, lon: float, start_date: str, end_date: str, timezone: str = 'auto') -> pd.DataFrame:
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "uv_index,uv_index_clear_sky,cloud_cover",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone
    }
    r = requests.get(base, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    tzname = js.get("timezone", "UTC")
    times = pd.to_datetime(js["hourly"]["time"])
    df = pd.DataFrame({
        "datetime_local": times.tz_localize(tzname),
        "UVI_all": js["hourly"]["uv_index"],
        "UVI_clear": js["hourly"]["uv_index_clear_sky"],
        "cloud_cover": js["hourly"]["cloud_cover"]
    })
    return df

def _list_local_hourly_files(data_dir: str) -> list:
    if not os.path.isdir(data_dir):
        return []
    files = []
    for name in os.listdir(data_dir):
        if name.startswith("hourly_") and name.endswith(".csv"):
            m = re.match(r"hourly_([\-0-9\.]+)_([\-0-9\.]+)_([0-9]{4}-[0-9]{2}-[0-9]{2})\.csv", name)
            if m:
                lat = float(m.group(1))
                lon = float(m.group(2))
                d = m.group(3)
                files.append({"path": os.path.join(data_dir, name), "lat": lat, "lon": lon, "date": d})
    return files

CITY_CATALOG = {
    "new york":    {"name": "New York, US",    "lat": 40.7128,  "lon": -74.0060,  "tz": "America/New_York"},
    "los angeles": {"name": "Los Angeles, US", "lat": 34.0522,  "lon": -118.2437, "tz": "America/Los_Angeles"},
    "chicago":     {"name": "Chicago, US",     "lat": 41.8781,  "lon": -87.6298,  "tz": "America/Chicago"},
    "toronto":     {"name": "Toronto, CA",     "lat": 43.6532,  "lon": -79.3832,  "tz": "America/Toronto"},
    "mexico city": {"name": "Mexico City, MX", "lat": 19.4326,  "lon": -99.1332,  "tz": "America/Mexico_City"},
    "sao paulo":   {"name": "Sao Paulo, BR",   "lat": -23.5505, "lon": -46.6333,  "tz": "America/Sao_Paulo"},
    "buenos aires": {"name": "Buenos Aires, AR","lat": -34.6037, "lon": -58.3816,  "tz": "America/Argentina/Buenos_Aires"},
    "london":      {"name": "London, UK",      "lat": 51.5074,  "lon": -0.1278,   "tz": "Europe/London"},
    "paris":       {"name": "Paris, FR",       "lat": 48.8566,  "lon": 2.3522,    "tz": "Europe/Paris"},
    "berlin":      {"name": "Berlin, DE",      "lat": 52.5200,  "lon": 13.4050,   "tz": "Europe/Berlin"},
    "madrid":      {"name": "Madrid, ES",      "lat": 40.4168,  "lon": -3.7038,   "tz": "Europe/Madrid"},
    "rome":        {"name": "Rome, IT",        "lat": 41.9028,  "lon": 12.4964,   "tz": "Europe/Rome"},
    "palermo":     {"name": "Palermo, IT",     "lat": 38.1157,  "lon": 13.3615,   "tz": "Europe/Rome"},
    "catania":     {"name": "Catania, IT",     "lat": 37.5079,  "lon": 15.0830,   "tz": "Europe/Rome"},
    "istanbul":    {"name": "Istanbul, TR",    "lat": 41.0082,  "lon": 28.9784,   "tz": "Europe/Istanbul"},
    "dubai":       {"name": "Dubai, AE",       "lat": 25.2048,  "lon": 55.2708,   "tz": "Asia/Dubai"},
    "cairo":       {"name": "Cairo, EG",       "lat": 30.0444,  "lon": 31.2357,   "tz": "Africa/Cairo"},
    "johannesburg": {"name": "Johannesburg, ZA","lat": -26.2041, "lon": 28.0473,   "tz": "Africa/Johannesburg"},
    "nairobi":     {"name": "Nairobi, KE",     "lat":  -1.2921, "lon": 36.8219,   "tz": "Africa/Nairobi"},
    "moscow":      {"name": "Moscow, RU",      "lat": 55.7558,  "lon": 37.6173,   "tz": "Europe/Moscow"},
    "tokyo":       {"name": "Tokyo, JP",       "lat": 35.6762,  "lon": 139.6503,  "tz": "Asia/Tokyo"},
    "shanghai":    {"name": "Shanghai, CN",    "lat": 31.2304,  "lon": 121.4737,  "tz": "Asia/Shanghai"},
    "singapore":   {"name": "Singapore, SG",   "lat": 1.3521,   "lon": 103.8198,  "tz": "Asia/Singapore"},
    "hong kong":   {"name": "Hong Kong, HK",   "lat": 22.3193,  "lon": 114.1694,  "tz": "Asia/Hong_Kong"},
    "mumbai":      {"name": "Mumbai, IN",      "lat": 19.0760,  "lon": 72.8777,   "tz": "Asia/Kolkata"},
    "sydney":      {"name": "Sydney, AU",      "lat": -33.8688, "lon": 151.2093,  "tz": "Australia/Sydney"},
    "melbourne":   {"name": "Melbourne, AU",   "lat": -37.8136, "lon": 144.9631,  "tz": "Australia/Melbourne"},
    "seoul":       {"name": "Seoul, KR",       "lat": 37.5665,  "lon": 126.9780,  "tz": "Asia/Seoul"},
    "beijing":     {"name": "Beijing, CN",     "lat": 39.9042,  "lon": 116.4074,  "tz": "Asia/Shanghai"},
}

def city_lookup(name: str):
    key = name.strip().lower()
    if key in CITY_CATALOG:
        return CITY_CATALOG[key]
    best = None
    for k, v in CITY_CATALOG.items():
        if key in k:
            best = v
            break
    return best

def load_hourly_from_local_db(lat: float, lon: float, date_str: str, data_dir: str = "data") -> pd.DataFrame:
    os.makedirs(data_dir, exist_ok=True)
    target = os.path.join(data_dir, f"hourly_{lat:.4f}_{lon:.4f}_{date_str}.csv")
    if os.path.exists(target):
        df = pd.read_csv(target, parse_dates=["datetime_local"])
        if df["datetime_local"].dt.tz is None:
            df["datetime_local"] = df["datetime_local"].dt.tz_localize("UTC")
        return df
    catalog = os.path.join(data_dir, "hourly_data.csv")
    if os.path.exists(catalog):
        df_all = pd.read_csv(catalog, parse_dates=["datetime_local"])
        mask = (
            (df_all["date"] == date_str)
            & (np.isclose(df_all["lat"], lat))
            & (np.isclose(df_all["lon"], lon))
        )
        df = df_all.loc[mask, ["datetime_local", "UVI_all", "UVI_clear"]].copy()
        if not df.empty:
            if df["datetime_local"].dt.tz is None:
                df["datetime_local"] = df["datetime_local"].dt.tz_localize("UTC")
            return df
    alt = os.path.join("outputs", f"uv_minute_{lat:.4f}_{lon:.4f}_{date_str}.csv")
    if os.path.exists(alt):
        mm = pd.read_csv(alt, parse_dates=["datetime_local"])
        mm = mm.set_index("datetime_local")
        hourly_all = mm["UVI_all_sky"].resample("1h").mean()
        hourly_clear = mm["UVI_clear_sky"].resample("1h").mean()
        out = pd.DataFrame({
            "datetime_local": hourly_all.index,
            "UVI_all": hourly_all.values,
            "UVI_clear": hourly_clear.values,
        })
        out.to_csv(target, index=False)
        out["datetime_local"] = pd.to_datetime(out["datetime_local"])  
        if out["datetime_local"].dt.tz is None:
            out["datetime_local"] = out["datetime_local"].dt.tz_localize("UTC")
        return out
    raise FileNotFoundError("No local hourly dataset found for the requested location and date")

def save_hourly_to_local_db(df: pd.DataFrame, lat: float, lon: float, date_str: str, data_dir: str = "data") -> str:
    os.makedirs(data_dir, exist_ok=True)
    target = os.path.join(data_dir, f"hourly_{lat:.4f}_{lon:.4f}_{date_str}.csv")
    cols = ["datetime_local", "UVI_all", "UVI_clear"]
    if not all(c in df.columns for c in cols):
        tmp = pd.DataFrame({
            "datetime_local": df["datetime_local"],
            "UVI_all": df["UVI_all"],
            "UVI_clear": df["UVI_clear"],
        })
    else:
        tmp = df[cols].copy()
    tmp.to_csv(target, index=False)
    return target

def minute_engine(lat: float, lon: float, date_str: str, timezone: str = 'auto', p_mu: float = 1.2, hourly_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    dt = pd.to_datetime(date_str).date()
    hourly = hourly_df if hourly_df is not None else fetch_open_meteo_uvi(lat, lon, start_date=date_str, end_date=date_str, timezone=timezone)
    tzobj = hourly["datetime_local"].dt.tz if hasattr(hourly["datetime_local"].dt, "tz") else pytz.UTC
    if tzobj is None:
        tzobj = pytz.UTC
    day_start = pd.to_datetime(date_str).tz_localize(tzobj)
    minutes = pd.date_range(day_start, day_start + pd.Timedelta(days=1), freq="1min", tz=tzobj, inclusive="left")
    _, mu, _ = solar_position_series(lat, lon, minutes)
    mu = np.clip(np.array(mu), 0.0, None)

    uvi_clear_hourly = hourly.set_index("datetime_local")["UVI_clear"]

    uvi_clear_minute = np.zeros(len(minutes))
    p = p_mu
    for h in range(len(uvi_clear_hourly)):
        h_start = uvi_clear_hourly.index[h]
        h_end   = h_start + pd.Timedelta(hours=1)
        mask = (minutes >= h_start) & (minutes < h_end)
        mu_seg = mu[mask] ** p
        if mu_seg.sum() > 0:
            uvi_target = float(uvi_clear_hourly.iloc[h])
            scale = uvi_target * (len(mu_seg)) / mu_seg.sum()
            uvi_clear_minute[mask] = mu_seg * scale
        else:
            uvi_clear_minute[mask] = 0.0

    ratio_hourly = (hourly.set_index("datetime_local")["UVI_all"] / hourly.set_index("datetime_local")["UVI_clear"]).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.3)
    ratio_hourly = ratio_hourly.reindex(uvi_clear_hourly.index).interpolate(limit_direction="both")
    ratio_min = ratio_hourly.reindex(minutes).interpolate("time").bfill().ffill().to_numpy()

    uvi_all_minute = uvi_clear_minute * ratio_min

    df = pd.DataFrame({
        "datetime_local": minutes,
        "UVI_clear_sky": uvi_clear_minute,
        "UVI_all_sky": uvi_all_minute
    })
    df["E_ery_Wm2"] = df["UVI_all_sky"] / 40.0
    df["dose_Jm2"]  = (df["E_ery_Wm2"] * 60.0).cumsum()
    df["dose_SED"]  = df["dose_Jm2"] / 100.0
    return df

def validate_reaggregation(df_minute: pd.DataFrame, hourly_ref: pd.DataFrame, tzname: str, out_prefix: str=None):
    mh = df_minute.set_index("datetime_local")["UVI_all_sky"].resample("1h").mean()
    ref = hourly_ref.set_index("datetime_local")["UVI_all"]
    joined = pd.DataFrame({"minute_to_hour": mh, "api_hourly": ref}).dropna()
    err = joined["minute_to_hour"] - joined["api_hourly"]
    mae = float(err.abs().mean())
    rmse = float(np.sqrt((err**2).mean()))
    if out_prefix:
        plt.figure(figsize=(6,6))
        plt.scatter(joined["api_hourly"], joined["minute_to_hour"])
        lims = [0, max(1e-6, joined.max().max())]
        plt.plot(lims, lims)
        plt.xlabel("API hourly UVI"); plt.ylabel("Re-aggregated minute UVI"); plt.title("Validation: hourly vs minute")
        plt.tight_layout(); plt.savefig(f"{out_prefix}_scatter.png", dpi=150); plt.close()

        plt.figure(figsize=(10,4))
        plt.plot(joined.index, err)
        plt.xlabel("Local time"); plt.ylabel("Error (min→hour − API)"); plt.xticks(rotation=45); plt.title("Validation error by hour")
        plt.tight_layout(); plt.savefig(f"{out_prefix}_error_by_hour.png", dpi=150); plt.close()
    return {"MAE": mae, "RMSE": rmse, "n": int(len(joined))}

def prompt_interactive(existing_data: list) -> dict:
    print("Interactive mode")
    presets = {
        "1": {"name": "Los Angeles, US", "lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles"},
        "2": {"name": "Palermo, IT", "lat": 38.1157, "lon": 13.3615, "tz": "Europe/Rome"},
        "3": {"name": "Catania, IT", "lat": 37.5079, "lon": 15.0830, "tz": "Europe/Rome"},
    }
    print("Choose a preset location or press Enter to type values:")
    if existing_data:
        print("[D] Choose from local datasets")
    print("[C] Choose a city by name (e.g., Rome, London, Sydney)")
    for k, v in presets.items():
        print(f"[{k}] {v['name']} ({v['lat']:.4f},{v['lon']:.4f}) {v['tz']}")
    choice = input("Selection [Enter to skip]: ").strip()
    if existing_data and (choice.lower() == 'd'):
        for i, it in enumerate(existing_data, 1):
            print(f"[{i}] lat {it['lat']:.4f}, lon {it['lon']:.4f}, date {it['date']}")
        idx = input("Select a dataset by number: ").strip()
        if idx.isdigit() and 1 <= int(idx) <= len(existing_data):
            sel = existing_data[int(idx)-1]
            lat = sel['lat']
            lon = sel['lon']
            tzname = "auto"
            date_str = sel['date']
            outdir = input("Output directory [outputs]: ").strip() or "outputs"
            p_mu = input("Exponent p for mu^p [1.2]: ").strip()
            p_mu = float(p_mu) if p_mu else 1.2
            print("Data source set to Local")
            return {"lat": lat, "lon": lon, "tzname": tzname, "date": date_str, "outdir": outdir, "p_mu": p_mu, "source": "local"}
    if choice.lower() == 'c':
        print("Known cities:")
        names = list(CITY_CATALOG.keys())
        for i, k in enumerate(names, 1):
            v = CITY_CATALOG[k]
            print(f"[{i}] {v['name']} ({v['lat']:.4f},{v['lon']:.4f}) {v['tz']}")
        q = input("Type a city name or choose number: ").strip()
        selected = None
        if q.isdigit() and 1 <= int(q) <= len(names):
            selected = CITY_CATALOG[names[int(q)-1]]
        else:
            selected = city_lookup(q)
        if selected is not None:
            lat = selected["lat"]
            lon = selected["lon"]
            tzname = selected["tz"]
        else:
            print("City not found; defaulting to manual entry")
            lat = float(input("Latitude: ").strip())
            lon = float(input("Longitude: ").strip())
            tzname = input("Timezone (IANA, e.g. Europe/Rome or 'auto'): ").strip() or "auto"
        date_str = input("Date (YYYY-MM-DD, default today): ").strip() or str(date.today())
        outdir = input("Output directory [outputs]: ").strip() or "outputs"
        p_mu = input("Exponent p for mu^p [1.2]: ").strip()
        p_mu = float(p_mu) if p_mu else 1.2
        print("Data source: [L]ocal database, [A]PI, [Auto]")
        src = input("Choose source [Auto]: ").strip().lower() or "auto"
        if src.startswith("l"):
            source = "local"
        elif src.startswith("a"):
            source = "api"
        else:
            source = "auto"
        return {"lat": lat, "lon": lon, "tzname": tzname, "date": date_str, "outdir": outdir, "p_mu": p_mu, "source": source}
    if choice in presets:
        lat = presets[choice]["lat"]
        lon = presets[choice]["lon"]
        tzname = presets[choice]["tz"]
    else:
        lat = float(input("Latitude: ").strip())
        lon = float(input("Longitude: ").strip())
        tzname = input("Timezone (IANA, e.g. Europe/Rome or 'auto'): ").strip() or "auto"
    date_str = input("Date (YYYY-MM-DD, default today): ").strip() or str(date.today())
    outdir = input("Output directory [outputs]: ").strip() or "outputs"
    p_mu = input("Exponent p for mu^p [1.2]: ").strip()
    p_mu = float(p_mu) if p_mu else 1.2
    print("Data source: [L]ocal database, [A]PI, [Auto]")
    src = input("Choose source [Auto]: ").strip().lower() or "auto"
    if src.startswith("l"):
        source = "local"
    elif src.startswith("a"):
        source = "api"
    else:
        source = "auto"
    return {"lat": lat, "lon": lon, "tzname": tzname, "date": date_str, "outdir": outdir, "p_mu": p_mu, "source": source}

def get_hourly(lat: float, lon: float, date_str: str, timezone: str, source: str) -> Tuple[pd.DataFrame, str]:
    if source == "local":
        return load_hourly_from_local_db(lat, lon, date_str), "local"
    if source == "api":
        return fetch_open_meteo_uvi(lat, lon, start_date=date_str, end_date=date_str, timezone=timezone), "api"
    try:
        return load_hourly_from_local_db(lat, lon, date_str), "local"
    except Exception:
        return fetch_open_meteo_uvi(lat, lon, start_date=date_str, end_date=date_str, timezone=timezone), "api"

def _collect_cached_dates(lat: float, lon: float, data_dir: str = "data") -> list:
    dates = []
    if not os.path.isdir(data_dir):
        return dates
    prefix = f"hourly_{lat:.4f}_{lon:.4f}_"
    for name in os.listdir(data_dir):
        if name.startswith(prefix) and name.endswith('.csv'):
            dates.append(name.replace(prefix, '').replace('.csv',''))
    return sorted(dates)

def generate_world_map_html(out_path: str, catalog: dict):
    items = []
    for k, v in catalog.items():
        lat = float(v["lat"])
        lon = float(v["lon"])
        dates = _collect_cached_dates(lat, lon)
        items.append({
            "name": v["name"],
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "tz": v["tz"],
            "dates": dates,
        })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>")
        f.write("<title>Mengjie's UVI World Engine</title>")
        f.write("<link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" integrity=\"sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=\" crossorigin=\"\"/>")
        f.write("<style>html,body,#map{height:100%;margin:0;} .panel{position:absolute;left:10px;background:#fff;padding:8px 10px;border:1px solid #ddd;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,0.1);} .panel-title{top:10px;} .panel-help{top:70px;} .brand{font-size:16px;font-weight:600;color:#111} .by{font-size:12px;color:#444;margin-top:2px} .btn{display:inline-block;padding:4px 8px;border:1px solid #888;border-radius:4px;margin:2px 0;cursor:pointer;} .small{font-size:12px;color:#333} ul{margin:4px 0 0 14px;padding:0;} li{margin:0;padding:0;} </style>")
        f.write("</head><body>")
        f.write("<div id='map'></div>")
        f.write("<div class='panel panel-title'><div class='brand'>Mengjie's UVI World Engine</div><div class='by'>by Mengjie Fan</div></div>")
        f.write("<div class='panel panel-help small'>Click a city marker to see cached dates and a copyable CLI command.</div>")
        f.write("<script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\" integrity=\"sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=\" crossorigin=\"\"></script>")
        f.write("<script>const CITIES = "+json.dumps(items)+";\n")
        f.write("const map = L.map('map',{worldCopyJump:true}).setView([20,0], 2);\n")
        f.write("L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom: 8, attribution: '&copy; OpenStreetMap contributors'}).addTo(map);\n")
        f.write("function popupHtml(c){const lat=c.lat.toFixed(4),lon=c.lon.toFixed(4);let h=`<b>${c.name}</b><br><span class=small>${lat}, ${lon} — ${c.tz}</span>`; if(c.dates.length){h+=`<div class=small><b>Cached dates:</b><ul>`; c.dates.forEach(d=>{const outCsv=`outputs/uv_minute_${lat}_${lon}_${d}.csv`; h+=`<li>${d} — <a href='${outCsv}' target='_blank'>CSV</a> <span class=btn onclick=copyCmd('${lat}','${lon}','${d}')>Copy CLI</span></li>`}); h+='</ul></div>'; } else { h+=`<div class=small>No cached datasets. <span class=btn onclick=copyCmd('${lat}','${lon}', new Date().toISOString().slice(0,10))>Copy CLI for today</span></div>`;} return h;}\n")
        f.write("function copyCmd(lat,lon,date){const cmd=`python uv_minute_app.py --lat ${lat} --lon ${lon} --date ${date} --source api --outdir outputs --html`; navigator.clipboard.writeText(cmd).then(()=>{alert('Copied to clipboard:\n'+cmd);});}\n")
        f.write("CITIES.forEach(c=>{const m=L.marker([c.lat,c.lon]).addTo(map); m.bindPopup(popupHtml(c)); m.bindTooltip(c.name);});\n")
        f.write("</script></body></html>")

def main():
    ap = argparse.ArgumentParser(description="Minute-resolved UV engine with interactive and local data support")
    ap.add_argument("--lat", type=float, required=False, help="Latitude")
    ap.add_argument("--lon", type=float, required=False, help="Longitude")
    ap.add_argument("--date", type=str, required=False, help="Date in YYYY-MM-DD format")
    ap.add_argument("--timezone", type=str, default='auto', help="Timezone")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    ap.add_argument("--p_mu", type=float, default=1.2, help="Exponent for solar radiation calculation")
    ap.add_argument("--html", action="store_true", help="Generate interactive HTML report")
    ap.add_argument("--source", type=str, default="auto", choices=["auto", "local", "api"], help="Data source")
    ap.add_argument("--no_cache", action="store_true", help="Do not cache fetched hourly data to data/")
    ap.add_argument("--interactive", action="store_true", help="Prompt for inputs interactively")
    ap.add_argument("--city", type=str, required=False, help="City name (e.g., 'Rome', 'London') to autoload lat/lon/timezone")
    ap.add_argument("--worldmap", action="store_true", help="Generate a world map HTML with city markers from catalog and cached data")
    ap.add_argument("--map_out", type=str, default=os.path.join("outputs", "world_map.html"), help="Output path for world map HTML")
    args = ap.parse_args()

    if args.worldmap:
        generate_world_map_html(args.map_out, CITY_CATALOG)
        print(json.dumps({"world_map": args.map_out, "cities": len(CITY_CATALOG)}))
        return

    if not args.interactive and args.city and (args.lat is None or args.lon is None):
        info = city_lookup(args.city)
        if info:
            args.lat = info["lat"]
            args.lon = info["lon"]
            if args.timezone == 'auto':
                args.timezone = info["tz"]
    need_prompt = args.interactive or (args.lat is None or args.lon is None or args.date is None)
    if need_prompt:
        files = _list_local_hourly_files("data")
        params = prompt_interactive(files)
        args.lat = params["lat"]
        args.lon = params["lon"]
        args.timezone = params["tzname"]
        args.date = params["date"]
        args.outdir = params["outdir"]
        args.p_mu = params["p_mu"]
        args.source = params["source"]

    os.makedirs(args.outdir, exist_ok=True)
    try:
        hourly, src_used = get_hourly(args.lat, args.lon, args.date, args.timezone, args.source)
    except Exception as e:
        print(str(e))
        print("Falling back to local data if available")
        hourly = load_hourly_from_local_db(args.lat, args.lon, args.date)
        src_used = "local"
    if src_used == "api" and not args.no_cache:
        save_hourly_to_local_db(hourly, args.lat, args.lon, args.date)
    tzobj = hourly["datetime_local"].dt.tz if hasattr(hourly["datetime_local"].dt, "tz") else pytz.UTC
    tzname = str(tzobj)
    df = minute_engine(args.lat, args.lon, args.date, args.timezone, p_mu=args.p_mu, hourly_df=hourly)
    csv_path = os.path.join(args.outdir, f"uv_minute_{args.lat:.4f}_{args.lon:.4f}_{args.date}.csv")
    df.to_csv(csv_path, index=False)
    val = validate_reaggregation(df, hourly, tzname, out_prefix=os.path.join(args.outdir, "validation"))
    with open(os.path.join(args.outdir, "validation.json"), "w") as f:
        json.dump(val, f, indent=2)
    plt.figure(figsize=(10,4))
    plt.plot(df["datetime_local"], df["UVI_clear_sky"], label="Clear-sky UVI")
    plt.plot(df["datetime_local"], df["UVI_all_sky"], label="All-sky UVI")
    plt.legend(); plt.xlabel("Local time"); plt.ylabel("UV Index"); plt.xticks(rotation=45); plt.title("Minute UVI — clear vs all-sky")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "uvi_minute.png"), dpi=150); plt.close()
    plt.figure(figsize=(10,4))
    plt.plot(df["datetime_local"], df["dose_SED"])
    plt.xlabel("Local time"); plt.ylabel("Dose (SED)"); plt.xticks(rotation=45); plt.title("Cumulative erythemal dose (SED)")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "dose_sed.png"), dpi=150); plt.close()
    if args.html:
        html_path = os.path.join(args.outdir, "interactive.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>")
            f.write("<title>Mengjie's UVI World Engine</title>")
            f.write("<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;} .header{padding:14px 20px;border-bottom:1px solid #eee;background:#fafafa} .brand{font-weight:600;font-size:18px} .by{font-size:12px;color:#555;margin-top:2px} .content{padding:20px;} img{max-width:100%;height:auto;border:1px solid #ddd} h2{margin:10px 0 6px 0;font-size:16px}</style></head><body>")
            f.write("<div class='header'><div class='brand'>Mengjie's UVI World Engine</div><div class='by'>by Mengjie Fan</div></div>")
            f.write("<div class='content'>")
            f.write(f"<h2>Run summary — {args.date}</h2>")
            f.write(f"<p><b>Location:</b> lat {args.lat:.4f}, lon {args.lon:.4f} — timezone {tzname}</p>")
            f.write(f"<p><a href='uvi_minute.png'>UVI (clear vs all-sky)</a> • <a href='dose_sed.png'>Cumulative dose (SED)</a> • <a href='{os.path.basename(csv_path)}'>Download CSV</a></p>")
            f.write("<p><a href='validation_scatter.png'>Validation scatter</a> • <a href='validation_error_by_hour.png'>Validation error by hour</a> • <a href='validation.json'>Validation metrics</a></p>")
            f.write("</div></body></html>")
    print(json.dumps({"csv": csv_path, "validation": val}, indent=2))

if __name__ == "__main__":
    main()
