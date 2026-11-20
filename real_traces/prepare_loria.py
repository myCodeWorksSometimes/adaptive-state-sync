"""
Convert the LORIA cloud-gaming QoS/QoE traces into the same row format as
dataset/network_dataset.csv, so the Phase 3 pipeline can run on real telemetry.

Source (not redistributed here; no licence is stated, so cite instead):
  https://cloud-gaming-traces.lhs.loria.fr/  -> "QoS/QoE features dataset" (7z, ~25 MB)
  Marchal et al., "An Analysis of Cloud Gaming Platforms Behavior under Different
  Network Constraints", JNSM 2023.

What is real and what is not
  real      : RTT, jitter, packet loss and link throughput, recorded via WebRTC while
              playing racing games on Stadia, GeForce Now and xCloud over emulated
              Orange 4G links (Mahimahi replay of real traces) and one FTTH link.
  synthetic : game-side context (entity_count, payload_bytes, n_clients), drawn with
              the same Racing ranges as generate_dataset.py, because the traces carry
              cloud-gaming video, not game-state synchronization.

Conversion choices
  * Rows (~every 15 ms) are binned to 60 Hz ticks (last value per tick, then ffill).
  * RTT ('rtts') is only refreshed by WebRTC every ~2.5 s; it is held between updates.
  * Jitter: WebRTC video jitter is reported in RTP clock units (90 kHz) -> /90 for ms.
  * Loss: per-tick loss rate from the cumulative packetsLost/packetsReceived counters.
  * Bandwidth: the 'throughput' column (emulated link capacity, Mbps). The FTTH file has
    no such column, so its received bitrate ('rx_bps', Mbps) is used as a lower bound.
  * Each continuous trace is cut into 600-tick (10 s) sessions; rows, rolling windows
    [t-20, t-1], severity rule and H=5 targets match generate_dataset.py exactly.

Usage (from the project root):
  python real_traces/prepare_loria.py            # downloads, extracts, converts
Output: real_traces/loria_dataset.csv (git-ignored; regenerate with this script)
"""
import glob
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

URL = ("https://cloud-gaming-traces.lhs.loria.fr/"
       "ANR-19-CE25-0012_std_gfn_xc_cg_webrtc_metrics.7z")
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "loria_dataset.csv")

SEED = 42
TICK_MS = 1000 / 60
TICKS_PER_SESSION = 600
WINDOW = 20
HORIZON = 5
RACING = dict(entity_min=4, entity_max=12, payload_min=100, payload_max=400)
CLIENT_COUNTS = [4, 8, 16, 32]


def severity_label(rtt, loss_rate_mean):
    """Identical to generate_dataset.severity_label."""
    if rtt < 50 and loss_rate_mean < 0.005:
        return 0
    if rtt < 100 and loss_rate_mean < 0.015:
        return 1
    if rtt < 180 and loss_rate_mean < 0.030:
        return 2
    return 3


def download_and_extract():
    os.makedirs(RAW_DIR, exist_ok=True)
    archive = os.path.join(RAW_DIR, "loria_qos.7z")
    if not os.path.exists(archive):
        print("Downloading", URL)
        urllib.request.urlretrieve(URL, archive)
    if not glob.glob(os.path.join(RAW_DIR, "*", "*.csv")):
        import py7zr  # pip install py7zr
        with py7zr.SevenZipFile(archive) as z:
            z.extractall(RAW_DIR)


def to_ticks(path):
    """Resample one CSV to a 60 Hz tick series of rtt, jitter, loss, bw."""
    d = pd.read_csv(path)
    bw_col = "throughput" if "throughput" in d.columns else "rx_bps"
    d = d[["time_ms", "rtts", "jitter", "packetsLost", "packetsReceived", bw_col]]
    d = d.apply(pd.to_numeric, errors="coerce").dropna(subset=["time_ms"])
    d = d.sort_values("time_ms")
    tick = np.floor((d["time_ms"] - d["time_ms"].iloc[0]) / TICK_MS).astype(int)
    g = d.groupby(tick).last().reindex(range(int(tick.max()) + 1)).ffill().dropna()
    lost = g["packetsLost"].diff().clip(lower=0)
    recv = g["packetsReceived"].diff().clip(lower=0)
    loss = (lost / (lost + recv).replace(0, np.nan)).fillna(0.0)
    return pd.DataFrame({
        "rtt": g["rtts"].clip(lower=1.0).values,
        "jitter": (g["jitter"] / 90.0).values,
        "loss": loss.values,
        "bw": g[bw_col].clip(lower=1.0).values,
    }), bw_col


def build_rows(trace, path_label, rng, first_rep):
    rows = []
    n_sessions = len(trace) // TICKS_PER_SESSION
    for s in range(n_sessions):
        seg = trace.iloc[s * TICKS_PER_SESSION:(s + 1) * TICKS_PER_SESSION]
        rtt, jit, loss, bw = (seg[c].values for c in ("rtt", "jitter", "loss", "bw"))
        n_clients = CLIENT_COUNTS[(first_rep + s) % len(CLIENT_COUNTS)]
        e_max = min(RACING["entity_max"], RACING["entity_min"] + n_clients * 2)
        entities = rng.integers(RACING["entity_min"], e_max, size=TICKS_PER_SESSION)
        for t in range(WINDOW, TICKS_PER_SESSION - HORIZON):
            w = slice(t - WINDOW, t)
            loss_now = float(np.mean(loss[w]))
            loss_fut = float(np.mean(loss[t + HORIZON - WINDOW:t + HORIZON]))
            rows.append({
                "path": path_label, "genre": "Racing", "n_clients": n_clients,
                "repetition": first_rep + s, "tick": t,
                "rtt_ms": round(rtt[t], 2), "jitter_ms": round(jit[t], 2),
                "loss_rate": round(loss_now, 4), "bandwidth_mbps": round(bw[t], 2),
                "entity_count": int(entities[t]),
                "payload_bytes": int(rng.integers(RACING["payload_min"], RACING["payload_max"])),
                "rtt_mean": round(float(np.mean(rtt[w])), 2),
                "rtt_std": round(float(np.std(rtt[w])), 2),
                "rtt_min": round(float(np.min(rtt[w])), 2),
                "rtt_max": round(float(np.max(rtt[w])), 2),
                "jitter_mean": round(float(np.mean(jit[w])), 2),
                "jitter_std": round(float(np.std(jit[w])), 2),
                "bw_mean": round(float(np.mean(bw[w])), 2),
                "bw_std": round(float(np.std(bw[w])), 2),
                "loss_rate_mean": round(loss_now, 4),
                "genre_encoded": 1,
                "severity": severity_label(rtt[t], loss_now),
                "future_rtt_ms_h5": round(rtt[t + HORIZON], 2),
                "future_bw_mbps_h5": round(bw[t + HORIZON], 2),
                "future_severity_h5": severity_label(rtt[t + HORIZON], loss_fut),
            })
    return rows, n_sessions


def main():
    download_and_extract()
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*", "*.csv")))
    if not files:
        sys.exit("No CSV files found under " + RAW_DIR)
    rng = np.random.default_rng(SEED)
    all_rows = []
    for f in files:
        platform = os.path.basename(os.path.dirname(f)).replace("racing_", "")  # std_1, gfn_1, xc_1
        condition = os.path.splitext(os.path.basename(f))[0]                  # excellent ... highway, ftth
        trace, bw_col = to_ticks(f)
        rows, n = build_rows(trace, f"{platform}__{condition}", rng, first_rep=0)
        all_rows.extend(rows)
        print(f"{platform:6s} {condition:10s} ticks={len(trace):7d} sessions={n:4d} bw_from={bw_col}")
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT, index=False)
    print(f"\nSaved {len(df):,} rows, {df.groupby(['path', 'repetition']).ngroups} sessions -> {OUT}")
    print("Current severity share (%):", (df["severity"].value_counts(normalize=True).sort_index() * 100).round(1).to_dict())


if __name__ == "__main__":
    main()
