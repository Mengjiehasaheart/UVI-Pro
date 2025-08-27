#!/usr/bin/env python3

__author__ = "Mengjie"

import argparse, sys, math, json, io, os
from dataclasses import dataclass
from typing import Dict, Tuple
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta
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
        "datetime_local": times.tz_localize("UTC").tz_convert(tzname),
        "UVI_all": js["hourly"]["uv_index"],
        "UVI_clear": js["hourly"]["uv_index_clear_sky"],
        "cloud_cover": js["hourly"]["cloud_cover"]
    })
    return df

def minute_engine(lat: float, lon: float, date_str: str, timezone: str = 'auto', p_mu: float = 1.2) -> pd.DataFrame:
    dt = pd.to_datetime(date_str).date()
    hourly = fetch_open_meteo_uvi(lat, lon, start_date=date_str, end_date=date_str, timezone=timezone)
    tzname = str(hourly["datetime_local"].dt.tz.zone)
    day_start = pd.to_datetime(date_str).tz_localize(tzname)
    minutes = pd.date_range(day_start, day_start + pd.Timedelta(days=1), freq="1min", tz=tzname, inclusive="left")
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

def main():
    ap = argparse.ArgumentParser(description="A script to calculate minute-by-minute UV data.")
    ap.add_argument("--lat", type=float, required=True, help="Latitude")
    ap.add_argument("--lon", type=float, required=True, help="Longitude")
    ap.add_argument("--date", type=str, required=True, help="Date in YYYY-MM-DD format")
    ap.add_argument("--timezone", type=str, default='auto', help="Timezone")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    ap.add_argument("--p_mu", type=float, default=1.2, help="Exponent for solar radiation calculation")
    ap.add_argument("--html", action="store_true", help="Generate interactive HTML report")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    hourly = fetch_open_meteo_uvi(args.lat, args.lon, args.date, args.date, args.timezone)
    tzname = str(hourly["datetime_local"].dt.tz.zone)
    df = minute_engine(args.lat, args.lon, args.date, args.timezone, p_mu=args.p_mu)
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
            f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>Minute UV</title>")
            f.write("<style>body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; }")
            f.write("img { max-width: 100%; height: auto; border: 1px solid #ddd; } h1 { margin: 0 0 10px 0; }</style></head><body>")
            f.write(f"<h1>Minute UV — {args.date}</h1>")
            f.write(f"<p><b>Location:</b> lat {args.lat:.4f}, lon {args.lon:.4f} — timezone {tzname}</p>")
            f.write(f"<p><a href='uvi_minute.png'>UVI (clear vs all-sky)</a> • <a href='dose_sed.png'>Cumulative dose (SED)</a> • <a href='{os.path.basename(csv_path)}'>Download CSV</a></p>")
            f.write("<p><a href='validation_scatter.png'>Validation scatter</a> • <a href='validation_error_by_hour.png'>Validation error by hour</a> • <a href='validation.json'>Validation metrics</a></p>")
            f.write("</body></html>")
    print(json.dumps({"csv": csv_path, "validation": val}, indent=2))

if __name__ == "__main__":
    main()