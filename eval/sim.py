#!/usr/bin/env python3
"""Vehicle models and Frenet-frame reference utilities (world frame)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class VehicleParams:
    wheelbase: float = 2.7
    lf: float = 1.35
    lr: float = 1.35
    mass: float = 1500.0
    iz: float = 2250.0
    cf: float = 80000.0
    cr: float = 80000.0
    mu: float = 0.9
    max_steer: float = 0.5
    max_accel: float = 1.5
    max_decel: float = 3.5
    max_steer_rate: float = 0.6
    length: float = 4.7
    width: float = 1.8
    dt: float = 0.1

    def clip_control(self, accel, steer):
        return (float(np.clip(accel, -self.max_decel, self.max_accel)),
                float(np.clip(steer, -self.max_steer, self.max_steer)))


@dataclass
class EgoState:
    x: float
    y: float
    yaw: float
    v: float
    steer: float = 0.0

    @property
    def xy(self):
        return np.array([self.x, self.y])

    def copy(self):
        return EgoState(self.x, self.y, self.yaw, self.v, self.steer)


def step_kinematic(state: EgoState, accel, steer, p: VehicleParams,
                   dt: float | None = None) -> EgoState:
    dt = p.dt if dt is None else dt
    accel, steer = p.clip_control(accel, steer)
    v = max(0.0, state.v + accel * dt)
    yaw = float(wrap_angle(state.yaw + v / p.wheelbase * np.tan(steer) * dt))
    return EgoState(state.x + v * np.cos(state.yaw) * dt,
                    state.y + v * np.sin(state.yaw) * dt,
                    yaw, v, steer)


class DynamicBicycle:
    """Single-track model with linear tires and a friction circle.

    Used as the plant. Controllers may plan on the simpler kinematic model,
    which keeps the closed loop exposed to a mild model mismatch.
    """

    def __init__(self, x, y, yaw, v, p: VehicleParams):
        self.p = p
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)
        self.vx = float(v)
        self.vy = 0.0
        self.r = 0.0
        self.steer = 0.0

    @property
    def state(self):
        return EgoState(self.x, self.y, self.yaw,
                        float(np.hypot(self.vx, self.vy)), self.steer)

    def step(self, accel, steer, dt: float | None = None):
        p = self.p
        dt = p.dt if dt is None else dt
        accel, steer = p.clip_control(accel, steer)
        dsteer = np.clip(steer - self.steer,
                         -p.max_steer_rate * dt, p.max_steer_rate * dt)
        self.steer = float(self.steer + dsteer)
        delta = self.steer

        if self.vx < 1.0:
            # Fall back to kinematics near standstill, where slip angles blow up.
            self.yaw = float(wrap_angle(self.yaw + self.vx / p.wheelbase
                                        * np.tan(delta) * dt))
            self.x += self.vx * np.cos(self.yaw) * dt
            self.y += self.vx * np.sin(self.yaw) * dt
            self.vx = max(0.0, self.vx + accel * dt)
            self.vy, self.r = 0.0, 0.0
            return self.state

        vx = max(self.vx, 0.5)
        a_f = delta - np.arctan2(self.vy + p.lf * self.r, vx)
        a_r = -np.arctan2(self.vy - p.lr * self.r, vx)
        fz_f = p.mass * G * p.lr / p.wheelbase
        fz_r = p.mass * G * p.lf / p.wheelbase
        fy_f = float(np.clip(p.cf * a_f, -p.mu * fz_f, p.mu * fz_f))
        fy_r = float(np.clip(p.cr * a_r, -p.mu * fz_r, p.mu * fz_r))

        ay = (fy_f * np.cos(delta) + fy_r) / p.mass
        # Longitudinal command is limited by what the friction circle leaves.
        ax_avail = np.sqrt(max(0.0, (p.mu * G) ** 2 - ay ** 2))
        ax = float(np.clip(accel, -min(p.max_decel, ax_avail),
                           min(p.max_accel, ax_avail)))

        dvx = ax + self.vy * self.r
        dvy = ay - self.vx * self.r
        dr = (p.lf * fy_f * np.cos(delta) - p.lr * fy_r) / p.iz

        self.x += (self.vx * np.cos(self.yaw) - self.vy * np.sin(self.yaw)) * dt
        self.y += (self.vx * np.sin(self.yaw) + self.vy * np.cos(self.yaw)) * dt
        self.yaw = float(wrap_angle(self.yaw + self.r * dt))
        self.vx = max(0.0, self.vx + dvx * dt)
        self.vy += dvy * dt
        self.r += dr * dt
        return self.state


def rollout_kinematic(x0, y0, yaw0, v0, accel, steer, p: VehicleParams,
                      dt: float | None = None):
    """Vectorised kinematic rollout.

    x0, y0, yaw0, v0 are scalars; accel and steer are (K, H) arrays.
    Returns x, y, yaw, v each (K, H).
    """
    dt = p.dt if dt is None else dt
    K, H = accel.shape
    x = np.empty((K, H))
    y = np.empty((K, H))
    yaw = np.empty((K, H))
    v = np.empty((K, H))
    cx, cy, cyaw, cv = x0, y0, yaw0, v0
    for k in range(H):
        cv = np.maximum(0.0, cv + accel[:, k] * dt)
        cx = cx + cv * np.cos(cyaw) * dt
        cy = cy + cv * np.sin(cyaw) * dt
        cyaw = wrap_angle(cyaw + cv / p.wheelbase * np.tan(steer[:, k]) * dt)
        x[:, k], y[:, k], yaw[:, k], v[:, k] = cx, cy, cyaw, cv
    return x, y, yaw, v


class ReferencePath:
    """Resampled world-frame polyline with s, heading and curvature."""

    def __init__(self, xy, spacing: float = 0.25):
        xy = np.asarray(xy, dtype=float)
        if len(xy) < 2:
            raise ValueError("reference path needs at least two points")
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)]
        keep = np.r_[True, np.diff(s) > 1e-6]
        s, xy = s[keep], xy[keep]
        self.s_raw = s
        self.xy_raw = xy
        n = max(2, int(np.ceil(s[-1] / spacing)) + 1)
        self.s = np.linspace(0.0, s[-1], n)
        self.xy = np.column_stack([np.interp(self.s, s, xy[:, 0]),
                                   np.interp(self.s, s, xy[:, 1])])
        d = np.diff(self.xy, axis=0)
        yaw = np.arctan2(d[:, 1], d[:, 0])
        self.yaw = np.r_[yaw, yaw[-1]]
        kappa = np.diff(np.unwrap(self.yaw)) / np.maximum(np.diff(self.s), 1e-6)
        self.kappa = np.r_[kappa, kappa[-1]]

    @property
    def length(self):
        return float(self.s[-1])

    def point(self, s):
        scalar = np.ndim(s) == 0
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.s[-1])
        out = np.column_stack([np.interp(s, self.s, self.xy[:, 0]),
                               np.interp(s, self.s, self.xy[:, 1])])
        return out[0] if scalar else out

    def heading(self, s):
        return np.interp(np.clip(s, 0.0, self.s[-1]), self.s, self.yaw)

    def curvature(self, s):
        return np.interp(np.clip(s, 0.0, self.s[-1]), self.s, self.kappa)

    def project(self, xy):
        """Return (s, d) for one or more world points."""
        q = np.atleast_2d(np.asarray(xy, dtype=float))
        d2 = ((q[:, None, :] - self.xy[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(d2, axis=1)
        off = q - self.xy[idx]
        yaw = self.yaw[idx]
        along = off[:, 0] * np.cos(yaw) + off[:, 1] * np.sin(yaw)
        lat = -off[:, 0] * np.sin(yaw) + off[:, 1] * np.cos(yaw)
        s = self.s[idx] + along
        if np.isscalar(xy) or np.asarray(xy).ndim == 1:
            return float(s[0]), float(lat[0])
        return s, lat

    def to_world(self, s, d):
        scalar = np.ndim(s) == 0 and np.ndim(d) == 0
        p = np.atleast_2d(self.point(s))
        yaw = np.atleast_1d(self.heading(s))
        d = np.atleast_1d(np.asarray(d, dtype=float))
        out = p + np.column_stack([-np.sin(yaw), np.cos(yaw)]) * d[:, None]
        return out[0] if scalar else out

    def subpath(self, s0, s1):
        m = (self.s >= s0) & (self.s <= s1)
        if m.sum() < 2:
            return ReferencePath(np.stack([self.point(s0), self.point(s1)]))
        return ReferencePath(self.xy[m])


def quintic_batch(p0, v0, a0, p1, v1, a1, horizon, t):
    """Boundary-value quintic for S samples, evaluated at times t.

    p0, v0, a0, p1, v1, a1 are (S,) arrays; t is (H,).
    Returns (H, S) position, velocity and acceleration.
    """
    t = np.asarray(t, dtype=float)
    T = float(horizon)
    b = np.column_stack([p0, v0 * T, a0 * T ** 2, p1, v1 * T, a1 * T ** 2])
    A = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                  [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                  [0.0, 0.0, 2.0, 6.0, 12.0, 20.0]])
    coef = np.linalg.solve(A, b.T)                       # (6, S)
    tau = np.clip(t / T, 0.0, 1.0)[:, None]
    P = (coef[0] + tau * (coef[1] + tau * (coef[2] + tau * (coef[3]
         + tau * (coef[4] + tau * coef[5])))))
    dP = (coef[1] + tau * (2 * coef[2] + tau * (3 * coef[3]
          + tau * (4 * coef[4] + tau * 5 * coef[5])))) / T
    ddP = (2 * coef[2] + tau * (6 * coef[3] + tau * (12 * coef[4]
           + tau * 20 * coef[5]))) / T ** 2
    return P, dP, ddP
