#!/usr/bin/env python3
"""Interacting-multiple-model prediction (constant velocity + coordinated turn)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Forecast:
    t: np.ndarray
    modes: list          # [(probability, xy (H,2), yaw (H,)), ...]
    source: str = "imm"

    def mean(self):
        xy = np.zeros_like(self.modes[0][1])
        for p, traj, _ in self.modes:
            xy += p * traj
        return xy

    def samples(self, inflation: float = 0.4):
        """Weighted mode trajectories with a growing radius per step."""
        return [(p, traj, 0.25 + inflation * self.t) for p, traj, _ in self.modes]


def _transition(dt, w, turn):
    if not turn or abs(w) < 1e-4:
        return np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                         [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    c, s = np.cos(w * dt), np.sin(w * dt)
    return np.array([[1, 0, s / w, -(1 - c) / w],
                     [0, 1, (1 - c) / w, s / w],
                     [0, 0, c, -s],
                     [0, 0, s, c]], dtype=float)


class _Model:
    """4-state linear KF: x, y, vx, vy; fixed turn rate for the CT mode."""

    def __init__(self, xy, vxy, w, dt, turn):
        self.x = np.array([xy[0], xy[1], vxy[0], vxy[1]], dtype=float)
        self.dt = dt
        self.turn = turn
        self.w = float(w)
        self.P = np.diag([0.25, 0.25, 1.0, 1.0])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.R = np.eye(2) * 0.09
        self.Q = np.diag([0.05, 0.05, 0.5, 0.5]) * dt

    def F(self, dt=None):
        return _transition(self.dt if dt is None else dt, self.w, self.turn)

    def predict(self):
        F = self.F()
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return float(np.exp(-0.5 * y @ np.linalg.solve(S, y))) + 1e-9

    def position(self):
        return self.x[:2]


class IMMPredictor:
    """Two-mode IMM; forecasts in world coordinates."""

    def __init__(self, stay=0.92):
        self.trans = np.array([[stay, 1 - stay], [1 - stay, stay]])

    def _turn_rate(self, t_hist, xy, yaw_hist):
        if yaw_hist is not None and len(yaw_hist) >= 2:
            yaw = np.unwrap(np.asarray(yaw_hist, dtype=float))
            span = max(float(t_hist[-1] - t_hist[max(0, len(t_hist) - 4)]), 1e-3)
            return float(np.clip((yaw[-1] - yaw[max(0, len(yaw) - 4)]) / span, -0.8, 0.8))
        if len(xy) >= 3:
            d = np.diff(xy[-3:], axis=0)
            psi = np.arctan2(d[:, 1], d[:, 0])
            dt = max(float(np.median(np.diff(t_hist))), 1e-3)
            return float(np.clip(np.mean(np.diff(psi)) / dt, -0.8, 0.8))
        return 0.0

    def forecast(self, t_hist, xy_hist, yaw_hist=None, speed_hist=None,
                 horizon=3.0, dt_pred=0.2):
        t_hist = np.asarray(t_hist, dtype=float)
        xy = np.asarray(xy_hist, dtype=float)
        if len(t_hist) < 2:
            return self._constant(xy[-1], 0.0, 0.0, horizon, dt_pred)
        dt = max(float(np.median(np.diff(t_hist))), 1e-3)
        vxy = np.diff(xy, axis=0)[-1] / max(t_hist[-1] - t_hist[-2], 1e-3)
        w = self._turn_rate(t_hist, xy, yaw_hist)

        models = [_Model(xy[0], vxy, 0.0, dt, turn=False),
                  _Model(xy[0], vxy, w, dt, turn=True)]
        mu = np.array([0.6, 0.4])

        for k in range(len(t_hist) - 1):
            cbar = self.trans.T @ mu
            mix = (self.trans * mu[:, None]) / np.maximum(cbar[None, :], 1e-9)
            mixed = []
            for j in range(2):
                xm = sum(mix[i, j] * models[i].x for i in range(2))
                Pm = np.zeros_like(models[j].P)
                for i in range(2):
                    d = models[i].x - xm
                    Pm += mix[i, j] * (models[i].P + np.outer(d, d))
                mixed.append((xm, Pm))
            for j in range(2):
                models[j].x, models[j].P = mixed[j]
                models[j].predict()
            lik = np.array([models[j].update(xy[k + 1]) for j in range(2)])
            mu = lik * cbar
            mu = mu / max(mu.sum(), 1e-12)

        # A clear yaw rate should not be out-voted by the CV mode's small
        # one-step error, so gate the turn mode on the measured rate.
        if abs(w) > 0.03:
            gate = 1.0 / (1.0 + np.exp(-(abs(w) - 0.03) * 25.0))
            mu = np.array([1.0 - gate, gate])

        H = max(1, int(round(horizon / dt_pred)))
        times = np.arange(1, H + 1) * dt_pred
        modes = []
        for j, m in enumerate(models):
            modes.append((float(mu[j]),) + self._rollout(m, times, dt_pred))
        return Forecast(t=times, modes=modes)

    def _rollout(self, model, times, dt_pred):
        traj = np.zeros((len(times), 2))
        yaws = np.zeros(len(times))
        s = model.x.copy()
        F = model.F(dt_pred)
        for i in range(len(times)):
            s = F @ s
            traj[i] = s[:2]
            yaws[i] = float(np.arctan2(s[3], s[2]))
        return traj, yaws

    def _constant(self, xy, vx, vy, horizon, dt_pred):
        H = max(1, int(round(horizon / dt_pred)))
        times = np.arange(1, H + 1) * dt_pred
        traj = np.column_stack([xy[0] + vx * times, xy[1] + vy * times])
        yaw = np.full(H, float(np.arctan2(vy, vx)) if np.hypot(vx, vy) > 1e-3 else 0.0)
        return Forecast(t=times, modes=[(1.0, traj, yaw)], source="cv")
