"""
src/baselines.py
================
Implements the four baseline synchronization strategies from Phase 2 Step 6.

Each baseline is documented with:
  - Original paper source / inspiration
  - What was actually implemented
  - Differences from the published paper
  - Decision logic

IMPORTANT: None of these baselines are claimed to be faithful reproductions
of the cited papers. They are inspired approximations suitable for simulation
in the absence of the original game engines, real network paths, and hardware.
"""

from src.adaptive_controller import (
    estimate_traffic_mbps, POLICY_TABLE, TICK_RATE_HZ, COMPRESSION_FACTORS
)
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE A — Static Full Synchronization
# ═══════════════════════════════════════════════════════════════════════════════

class BaselineA_StaticFullSync:
    """
    Static Full Synchronization — no adaptation.

    Design:
      UR=1.00 (60Hz), CL=0 (no compression), IR=1.00 (all entities)
      regardless of network conditions.

    This is the implicit baseline in most reviewed literature — a server
    that always sends full-resolution state at the maximum tick rate.

    Source:
      Implicit in all papers reviewed in Phases 1–2. Closest explicit
      description in Monaco et al. (CLAAP, 2025) as the no-adaptation baseline.

    Differences from published baselines:
      N/A — this is a conceptual baseline, not taken from a specific paper.
    """
    name = "Baseline_A_StaticFullSync"
    ur   = 1.00
    cl   = 0
    ir   = 1.00

    def decide(self, *args, **kwargs):
        return self.ur, self.cl, self.ir


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE B — Fixed-Rule Mitigation (ALC-inspired)
# ═══════════════════════════════════════════════════════════════════════════════

class BaselineB_FixedRuleALC:
    """
    Fixed-Rule Mitigation — inspired by Advanced Lag Compensation (ALC).

    Design:
      Applies a constant moderate throttling: UR=0.75, CL=1, IR=0.90.
      No adaptive component — same settings regardless of current conditions.

    Inspiration:
      Lee & Chang (2018), "Enhancing the experience of multiplayer shooter games via
      advanced lag compensation", Proc. ACM MMSys 2018, pp. 284-293.

    What Lee & Chang (2018) actually did:
      ALC extends Traditional Lag Compensation with a provisional-hit/confirmation
      protocol that only compensates for victims below a 50ms RTT threshold.
      Tested on 12 participants using Unreal Engine 4, 250ms injected delay.
      Achieved 94.1% fewer shot-behind-cover incidents.

    What we implement:
      A fixed moderate synchronization reduction (UR=0.75, CL=1, IR=0.90)
      that approximates a "conservative throttle always applied" policy.
      We DO NOT implement the actual hit-registration protocol of ALC,
      which requires a game engine.

    Differences:
      1. ALC is a SERVER-SIDE hit adjudication protocol; we approximate it
         as a fixed-bandwidth throttle.
      2. ALC uses a 50ms RTT threshold; we apply it universally.
      3. No actual hit or positional data exists in our simulation.

    Label: APPROXIMATION inspired by Lee & Chang (2018), not faithful reproduction.
    """
    name = "Baseline_B_FixedRuleALC"
    ur   = 0.75
    cl   = 1
    ir   = 0.90

    def decide(self, *args, **kwargs):
        return self.ur, self.cl, self.ir


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE C — Prediction Without Adaptive Action (CLAAP-inspired)
# ═══════════════════════════════════════════════════════════════════════════════

class BaselineC_PredictNoAct:
    """
    Prediction Without Adaptive Action — inspired by CLAAP's binary gate.

    Design:
      Uses the same trained predictor as the proposed ASS system, but applies
      only a binary pass/fail gate rather than the continuous three-lever
      controller. Specifically:
        - If predicted severity == 0 (None): use UR=1.00, CL=0, IR=1.00 (full sync)
        - If predicted severity >= 1 (any degradation): use UR=0.50, CL=1, IR=0.80

      This isolates the contribution of the CONTINUOUS, MULTI-LEVER controller
      (the proposed ASS) over binary on/off gating.

    Inspiration:
      Monaco et al. (2025), "Real-time latency prediction for cloud gaming
      applications" (CLAAP), Computer Networks 264, 111235. CLAAP's Game-State-Prediction module acts as a binary gate —
      it either sends game state or holds it based on latency prediction.

    What CLAAP actually did:
      RBFNN + BOCPD predicting RTT at sub-300μs cost; integrated with a
      Game-State-Prediction module via ZeroMQ; sustained ~60 FPS up to 80 clients.
      Validated on a single racing game genre on Google Stadia (now discontinued).

    What we implement:
      Binary gating using our trained predictor's severity output.
      We DO NOT replicate CLAAP's Game-State-Prediction module or ZeroMQ integration.

    Differences:
      1. CLAAP was validated on real Stadia traffic; we use synthetic simulation.
      2. CLAAP gates entire game-state frames; we approximate with UR/CL/IR settings.
      3. CLAAP's binary gate is per-frame; ours is per-tick.

    Label: APPROXIMATION inspired by Monaco et al. (2025), not faithful reproduction.
    """
    name = "Baseline_C_PredictNoAct"
    # Binary gate thresholds
    _GATE_ON  = (1.00, 0, 1.00)   # full sync when predicted severity == 0
    _GATE_OFF = (0.50, 1, 0.80)   # moderate reduction when any degradation predicted

    def decide(self, predicted_severity):
        """
        Apply binary gate based on predicted severity.

        Parameters
        ----------
        predicted_severity : int
            Severity predicted by the trained predictor (0–3).

        Returns
        -------
        tuple : (ur, cl, ir)
        """
        if int(predicted_severity) == 0:
            return self._GATE_ON
        else:
            return self._GATE_OFF


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE D — Static Resource Placement (García et al.-inspired)
# ═══════════════════════════════════════════════════════════════════════════════

class BaselineD_StaticPlacement:
    """
    Static Resource Placement — inspired by García Fernández de Castro et al.'s
    fog/edge load balancing.

    Design:
      Applies a fixed moderate sync reduction that approximates the savings
      from optimal server placement without per-client adaptive synchronization.
      UR=0.85, CL=0, IR=0.95.

      The idea: placing edge servers closer to players reduces average RTT,
      which can be approximated as a fixed reduction in traffic demand.

    Inspiration:
      García Fernández de Castro et al. (2025), "Dynamic low-latency load balancing
      model to improve quality of experience in a hybrid fog and edge architecture
      for MMO games", Applied Sciences 15(12), 6379 — simulation achieving a
      67.5% reduction in average latency.

    What García et al. actually did:
      Fully synthetic simulation of edge server placement; no ML prediction;
      no client-sync adaptation; measured average latency and response time.

    What we implement:
      A fixed moderate sync policy (UR=0.85, CL=0, IR=0.95) representing
      the marginal bandwidth savings achievable from infrastructure placement
      without any per-client adaptation. This is a VERY rough approximation.

    Differences:
      1. García et al. study server placement geography; we vary sync settings.
      2. No actual edge server or geographic routing is simulated.
      3. The fixed UR=0.85 is not derived from placement calculations.

    Label: APPROXIMATION inspired by García Fernández de Castro et al. (2025), not faithful reproduction.
    """
    name = "Baseline_D_StaticPlacement"
    ur   = 0.85
    cl   = 0
    ir   = 0.95

    def decide(self, *args, **kwargs):
        return self.ur, self.cl, self.ir


# ═══════════════════════════════════════════════════════════════════════════════
# Utility: get all baselines
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_baselines():
    """Return dict of all baseline instances keyed by name."""
    return {
        BaselineA_StaticFullSync.name: BaselineA_StaticFullSync(),
        BaselineB_FixedRuleALC.name:   BaselineB_FixedRuleALC(),
        BaselineC_PredictNoAct.name:   BaselineC_PredictNoAct(),
        BaselineD_StaticPlacement.name: BaselineD_StaticPlacement(),
    }
