#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
forecast_pipeline_clcv.py

– Télécharge les inputs depuis Google Drive via gdown
– Extrait les coefficients calendaires pour CLCV
– Exporte dans output/ :
     • coefficients_clcv.csv       (format long)
     • coefficients_clcv_wide.csv  (format wide)
     • metrics_clcv.csv            (métriques performance)
"""
import os
import logging
from pathlib import Path

import gdown
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# ───────── CONFIG ────────────────
INPUT_FILES = {
    # votre fichier historique CLCV
    "historique_corrige_clcv.csv": "1AQJ4c42dSqJydpxrmwurwE_L970xY6OJ",
    "Calendrier.csv":               "1X-szdCNtW62MWyOpn21wDkxinx_1Ijp9",
    "Paye_calendrier.csv":          "1yjAHMX7h3U66Tx37rgBIK2z7OWyxRGKO",
}
UPLOAD_FOLDER_ID = "1kmOkjE_1BTaHhO5Ir0salPt8AOvYEo2x"

WORKDIR    = Path(".")
OUTPUT_DIR = WORKDIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

HIST_FILE  = WORKDIR / "historique_corrige_clcv.csv"
CAL_FILE   = WORKDIR / "Calendrier.csv"
PAYE_FILE  = WORKDIR / "Paye_calendrier.csv"

OUT_COEF   = OUTPUT_DIR / "coefficients_clcv.csv"
OUT_COEF_W = OUTPUT_DIR / "coefficients_clcv_wide.csv"
OUT_METRICS= OUTPUT_DIR / "metrics_clcv.csv"

SELECT_SITES = [
    "CLCV_Venissieux",
    "CLCV_Nantes",
    "CLCV_Houplines",
    "CLCV_Rungis",
    "CLCV_Vitry",
    "CLCV_VLG",
    "CLCV_Castries",
]

# Mapping sites → zone scolaire (CLCV)
ZONE_MAP = {
    "CLCV_Venissieux": "A",
    "CLCV_Nantes":     "B",
    "CLCV_Houplines":  "B",
    "CLCV_Rungis":     "C",
    "CLCV_Vitry":      "C",
    "CLCV_VLG":        "C",
    "CLCV_Castries":   "C",
}

# ───────── HEADLESS AUTH ────────────────
# charge mycreds.txt si fourni
if os.getenv("GDRIVE_MYCREDS"):
    with open("mycreds.txt", "w") as f:
        f.write(os.environ["GDRIVE_MYCREDS"])

def get_drive() -> GoogleDrive:
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.access_token_expired:
        gauth.Refresh()
    gauth.SaveCredentialsFile("mycreds.txt")
    return GoogleDrive(gauth)

# ───────── UTIL ────────────────
def _yearweek(date):
    iso = date.isocalendar()
    return f"{iso.year}-{iso.week:02d}"

def download_inputs():
    for name, fid in INPUT_FILES.items():
        out = WORKDIR / name
        url = f"https://drive.google.com/uc?id={fid}"
        logging.info(f"Downloading {name}")
        gdown.download(url, str(out), quiet=False)

# ───────── LOAD & PREPARE ────────────────
def load_clcv_history() -> pd.DataFrame:
    df = pd.read_csv(
        HIST_FILE,
        sep=";", encoding="latin-1", dayfirst=True,
        parse_dates=["DATE_LIVRAISON"],
        usecols=["DATE_LIVRAISON","NOM_SITE_PREP","POTENTIEL_CDE_PAR_CP_CORRIGE"],
        low_memory=False,
    ).rename(columns={
        "DATE_LIVRAISON": "date",
        "NOM_SITE_PREP":   "site",
        "POTENTIEL_CDE_PAR_CP_CORRIGE": "commandes",
    })
    df["commandes"] = (
        df["commandes"].astype(str)
          .str.replace(",", ".")
          .pipe(pd.to_numeric, errors="coerce")
          .fillna(0)
          .astype(int)
    )
    df["ID_SEM"] = df["date"].apply(_yearweek)
    # on agrège au cas où plusieurs CP par site/semaine
    return df.groupby(["site","ID_SEM"], as_index=False)[["commandes"]].sum()

def load_calendar() -> pd.DataFrame:
    cal = pd.read_csv(
        CAL_FILE,
        sep=";", encoding="latin-1", dayfirst=True,
        parse_dates=["JOUR"],
        usecols=[
            "JOUR","SEMAINE","TYPE_SEM_ZONE",
            "SEM_FERIE","SEM_PRE_FERIE","SEM_POST_FERIE","TYPE_SEM_FERIE",
        ],
        low_memory=False,
    ).rename(columns={"JOUR":"date","SEMAINE":"week"})
    for c in ["SEM_FERIE","SEM_PRE_FERIE","SEM_POST_FERIE"]:
        cal[c] = cal[c].map({"VRAI":1,"FAUX":0}).fillna(0).astype(int)
    paye = pd.read_csv(
        PAYE_FILE,
        sep=";", encoding="latin-1", dayfirst=True,
        parse_dates=["JOUR"],
        usecols=["JOUR","TYPE_SEM_PAYE_FCT"],
        low_memory=False,
    ).rename(columns={"JOUR":"date"})
    paye["TYPE_SEM_PAYE_FCT"] = paye["TYPE_SEM_PAYE_FCT"].fillna("S_NORMALE")
    cal = cal.merge(paye, on="date", how="left")
    cal["TYPE_SEM_PAYE_FCT"] = cal["TYPE_SEM_PAYE_FCT"].fillna("S_NORMALE")
    cal["ID_SEM"] = cal["date"].apply(_yearweek)
    return cal[[
        "ID_SEM","TYPE_SEM_ZONE",
        "SEM_FERIE","SEM_PRE_FERIE","SEM_POST_FERIE",
        "TYPE_SEM_FERIE","TYPE_SEM_PAYE_FCT"
    ]]

def build_clcv_dataset() -> pd.DataFrame:
    hist = load_clcv_history()
    cal  = load_calendar()
    df   = hist.merge(cal, on="ID_SEM", how="left")
    df["ZONE_SCOLAIRE"] = df["site"].map(ZONE_MAP).fillna("C")
    return df

# ───────── PIPELINE ────────────────
def run_pipeline():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Starting CLCV pipeline")

    # 1) Download inputs
    download_inputs()

    # 2) Build dataset
    df = build_clcv_dataset()

    # 3) Variables calendaires
    vars_cal = [
        "TYPE_SEM_ZONE",
        "SEM_FERIE","SEM_PRE_FERIE","SEM_POST_FERIE",
        "TYPE_SEM_FERIE","TYPE_SEM_PAYE_FCT",
    ]

    # 4) Liste des sites à conserver
    SELECT_SITES = [
        "CLCV_Venissieux",
        "CLCV_Nantes",
        "CLCV_Houplines",
        "CLCV_Rungis",
        "CLCV_Vitry",
        "CLCV_VLG",
        "CLCV_Castries",
    ]

    metrics = []
    results = []

    # 5) Boucle par site
    for site, grp in df.groupby("site"):
        grp = grp.sort_values("ID_SEM").reset_index(drop=True)
        grp["t"] = np.arange(len(grp))

        # détendage
        det = sm.OLS(grp["commandes"].astype(float), sm.add_constant(grp["t"])).fit()
        y_det = (grp["commandes"] - det.predict(sm.add_constant(grp["t"]))).astype(float)

        # régression multiple
        X = pd.get_dummies(grp[vars_cal], drop_first=True, dtype=float)
        X = sm.add_constant(X)
        mod = sm.OLS(y_det, X).fit()

        # prédiction complète + métriques
        y_pred = det.predict(sm.add_constant(grp["t"])) + mod.predict(X)
        y_true = grp["commandes"]
        mask24 = grp["ID_SEM"].str.startswith("2024-")
        yt24, yp24 = y_true[mask24], y_pred[mask24]

        metrics.append({
            "site": site,
            "r2_insample":   r2_score(y_true, y_pred),
            "mae_insample":  mean_absolute_error(y_true, y_pred),
            "mape_insample": mean_absolute_percentage_error(y_true, y_pred),
            "r2_2024":       (r2_score(yt24, yp24) if len(yt24)>0 else np.nan),
            "mae_2024":      (mean_absolute_error(yt24, yp24) if len(yt24)>0 else np.nan),
            "mape_2024":     (mean_absolute_percentage_error(yt24, yp24) if len(yt24)>0 else np.nan),
        })

        coefs = mod.params.reset_index()
        coefs.columns = ["variable", "coef"]
        coefs["site"] = site
        results.append(coefs)

    # 6) Export long (filtré)
    df_coefs = pd.concat(results, ignore_index=True)[["site","variable","coef"]]
    df_coefs = df_coefs[df_coefs["site"].isin(SELECT_SITES)]
    df_coefs.to_csv(OUT_COEF, sep=";", decimal=",", encoding="latin-1", index=False)

    # 7) Export wide (à partir du long déjà filtré)
    df_wide = df_coefs.pivot(index="site", columns="variable", values="coef").reset_index()
    df_wide.to_csv(OUT_COEF_W, sep=";", decimal=",", encoding="latin-1", index=False)

    # 8) Export metrics (on peut aussi filtrer si besoin)
    df_metrics = pd.DataFrame(metrics)
    df_metrics = df_metrics[df_metrics["site"].isin(SELECT_SITES)]
    df_metrics.to_csv(OUT_METRICS, sep=";", decimal=",", encoding="latin-1", index=False)

    logging.info("✅ CLCV pipeline complete, filtered files generated in 'output/'")

    # 9) Upload headless
    drive = get_drive()
    for path in [OUT_COEF, OUT_COEF_W, OUT_METRICS]:
        logging.info(f"Uploading {path.name}")
        f = drive.CreateFile({
            "title": path.name,
            "parents": [{"id": UPLOAD_FOLDER_ID}]
        })
        f.SetContentFile(str(path))
        f.Upload()
        logging.info(f"Uploaded → {path.name}")

        logging.info(f"Uploading {path.name}")
        f = drive.CreateFile({
            "title": path.name,
            "parents": [{"id": UPLOAD_FOLDER_ID}]
        })
        f.SetContentFile(str(path))
        f.Upload()
        logging.info(f"Uploaded → {path.name}")

if __name__ == "__main__":
    run_pipeline()
