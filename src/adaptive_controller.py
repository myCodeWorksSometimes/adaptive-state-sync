"""
src/adaptive_controller.py
===========================
Implements the Adaptive State Synchronization (ASS) Controller from Phase 2.

The controller maps predicted network conditions to three independently
adjustable synchronization levers:

  1. Update-Rate Scaling (UR): fraction of full 60Hz tick rate delivered
  2. Payload Compression / Selective Sync (CL): compression level 0–3
  3. Interest-Radius Adjustment (IR): fraction of full entity set replicated

Decision policy:
  Monotonic threshold-based policy on predicted severity (0–3).
  Higher predicted severity → lower UR, higher CL, lower IR.

Bandwidth model (corrected):
  estimated_traffic_bps = payload_bytes × 8 × compression_factor[CL] × UR × tick_rate × n_clients
  This yields total server-to-all-clients traffic in bits per second.
  Conversion to Mbps: divide by 1e6.

  Bandwidth reduction is computed RELATIVE TO BASELINE A (static full sync),
  not relative to available capacity. Available capacity is NOT used to cap
  traffic (unless packet loss is modelled, which we do NOT claim here).

Traffic assumptions (all documented):
  - Tick rate: 60 Hz
  - Protocol overhead: +28 bytes per packet (IP+UDP header)
  - Each tick produces one packet per client
  - Compression reduces payload bytes by the compression_factor
  - Interest radius reduction scales entity count proportionally
  - Retransmissions are NOT modelled (beyond scope)

Disclaimer on gameplay metrics:
  The controller estimates synchronization TRAFFIC (a network metric).
  It does NOT measure game-engine outputs like FPS, hit registration,
  or player positions. These require an actual game engine.
  Simulated "synchronization quality" proxies are labelled as SIMULATED
  and explained in Section 8 of the notebook.
"""

import numpy as np

# ── Controller constants ───────────────────────────────────────────────────────
TICK_RATE_HZ       = 60
PROTOCOL_OVERHEAD  = 28   # bytes per packet (IP 20 + UDP 8)

# Compression factors: fraction of payload bytes remaining after compression
COMPRESSION_FACTORS = {0: 1.00, 1: 0.80, 2: 0.55, 3: 0.35}

# Policy table: severity → (UR multiplier, CL level, IR fraction)
# UR: 1.00=60Hz, 0.75=45Hz, 0.50=30Hz, 0.25=15Hz
# IR: 1.00=all entities, 0.80=80%, 0.60=60%, 0.40=40%
POLICY_TABLE = {
    0: (1.00, 0, 1.00),   # None:     full rate, no compression, full radius
    1: (0.75, 1, 0.80),   # Mild:     45Hz, light compression, 80% radius
    2: (0.50, 2, 0.60),   # Moderate: 30Hz, medium compression, 60% radius
    3: (0.25, 3, 0.40),   # Severe:   15Hz, heavy compression, 40% radius
}


def controller_decision(predicted_severity, disable_ur=False,
                        disable_pl=False, disable_im=False):
    """
    Map predicted severity to synchronization control parameters.

    Parameters
    ----------
    predicted_severity : int
        Predicted network severity class (0=None, 1=Mild, 2=Moderate, 3=Severe).
    disable_ur : bool
        If True, disable update-rate scaling (fix UR=1.00). Used in ablation.
    disable_pl : bool
        If True, disable payload compression (fix CL=0). Used in ablation.
    disable_im : bool
        If True, disable interest-radius adjustment (fix IR=1.00). Used in ablation.

    Returns
    -------
    tuple : (ur_multiplier, compression_level, interest_radius_fraction)
    """
    # Clamp severity to valid range
    sev = max(0, min(3, int(predicted_severity)))
    ur, cl, ir = POLICY_TABLE[sev]

    if disable_ur: ur = 1.00
    if disable_pl: cl = 0
    if disable_im: ir = 1.00

    return ur, cl, ir


def estimate_traffic_mbps(payload_bytes, ur, cl, ir, entity_count,
                           n_clients, tick_rate=TICK_RATE_HZ):
    """
    Estimate server-to-all-clients synchronization traffic in Mbps.

    Model assumptions (documented):
      - Each tick, the server sends one packet per client
      - Packet payload = payload_bytes × compression_factor[CL]
      - Interest radius reduces replicated entity count proportionally
      - Protocol overhead is added per packet (28 bytes for IP+UDP)
      - Effective entity payload scales with IR
      - Total traffic = per_client_bytes × n_clients × tick_rate × UR

    Parameters
    ----------
    payload_bytes : float or int
        Base payload size in bytes (at full IR and no compression).
    ur : float
        Update-rate multiplier (0.25–1.00).
    cl : int
        Compression level (0–3).
    ir : float
        Interest-radius fraction (0.40–1.00).
    entity_count : int
        Current number of entities in the game state.
    n_clients : int
        Number of connected clients.
    tick_rate : int
        Server tick rate in Hz.

    Returns
    -------
    float : estimated total traffic in Mbps
    """
    comp = COMPRESSION_FACTORS[cl]

    # Payload adjusted for IR (fewer entities → less data) and compression
    effective_payload = payload_bytes * ir * comp

    # Bytes per packet = effective payload + protocol overhead
    bytes_per_packet = effective_payload + PROTOCOL_OVERHEAD

    # Traffic per second = bytes/packet × n_clients × tick_rate × UR
    bits_per_second = bytes_per_packet * 8 * n_clients * tick_rate * ur

    return bits_per_second / 1e6   # convert to Mbps


def compute_bandwidth_reduction(traffic_proposed_mbps, traffic_baseline_mbps):
    """
    Compute percentage reduction in synchronization traffic relative to baseline.

    Parameters
    ----------
    traffic_proposed_mbps : float
        Estimated traffic for the proposed/experimental method.
    traffic_baseline_mbps : float
        Estimated traffic for the full-synchronization baseline (Baseline A).

    Returns
    -------
    float : percentage reduction (positive = less traffic than baseline)
    """
    if traffic_baseline_mbps <= 0:
        return 0.0
    return 100.0 * (traffic_baseline_mbps - traffic_proposed_mbps) / traffic_baseline_mbps


def compute_sync_quality(rtt_ms, ur, ir, jitter_ms):
    """
    SIMULATED ESTIMATE of synchronization quality degradation.

    This is a simplified proxy metric, NOT a game-engine measurement.
    It represents the expected state divergence caused by reduced update
    frequency and entity culling.

    Assumptions:
      - Higher RTT increases state age at client
      - Lower UR means updates arrive less frequently (larger gap between refreshes)
      - Lower IR means some entities are not synchronized (missing state)
      - Jitter adds noise to effective state age

    The metric is dimensionless. It is RELATIVE — lower is better.
    It should NOT be compared to real game-engine reconciliation errors.

    Label: SIMULATED PROXY — not a measured game-engine output.
    """
    # State age approximation: average age of a state update given tick interval
    tick_interval_ms = 1000.0 / (TICK_RATE_HZ * max(ur, 0.01))
    state_age_ms     = rtt_ms + tick_interval_ms / 2.0

    # IR penalty: missing entities contribute proportionally
    coverage_penalty = 1.0 / max(ir, 0.01)

    # Jitter noise term
    jitter_noise = abs(np.random.normal(0, jitter_ms * 0.05 + 0.1))

    return max(0.0, state_age_ms * coverage_penalty * 0.01 + jitter_noise)


def compute_update_rate_hz(ur):
    """Return achieved update rate in Hz."""
    return TICK_RATE_HZ * ur


def describe_decision(sev, ur, cl, ir):
    """Return a human-readable description of the controller decision."""
    return (
        f"Severity={sev} → "
        f"UR={ur:.2f} ({ur*TICK_RATE_HZ:.0f}Hz), "
        f"CL={cl} (×{COMPRESSION_FACTORS[cl]:.2f} compression), "
        f"IR={ir:.2f} ({ir*100:.0f}% entities)"
    )
