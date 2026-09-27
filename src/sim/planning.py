import json
import math
from dataclasses import dataclass
from pathlib import Path

import grpc
import numpy as np
from scipy.interpolate import CubicSpline

from geometry import quat_wxyz, rot_axis, smooth


class NoRoom(Exception):
    pass


@dataclass
class Plan:
    q: np.ndarray
    hand: np.ndarray
    labels: list
    stages: list

    def __len__(self):
        return len(self.q)

    def first(self, label):
        return self.labels.index(label)


class Timeline:
    def __init__(self, robot, settings, q0, hand0):
        self.robot = robot
        self.s = settings
        self.dt = settings.fluid.dt_sub
        self.q = [np.array(q0, float)[None]]
        self.hand = [np.array(hand0, float)[None]]
        self.labels = ["start"]
        self.stages = []

    @property
    def last(self):
        return self.q[-1][-1].copy()

    def h(self, name):
        return float(self.hand[-1][-1][self.robot.hand_slides.index(name)])

    def _add(self, q, label, hand=None):
        self.q.append(q)
        self.hand.append(np.repeat(self.hand[-1][-1:], len(q), 0) if hand is None else hand)
        self.labels.extend([label] * len(q))

    def hold(self, seconds, label):
        self._add(np.repeat(self.last[None], max(1, int(round(seconds / self.dt))), 0), label)

    def hand_to(self, seconds, label, **goals):
        n = max(1, int(round(seconds / self.dt)))
        w = np.array([smooth((i + 1) / n) for i in range(n)])[:, None]
        h0 = self.hand[-1][-1].copy()
        h1 = h0.copy()
        for k, v in goals.items():
            h1[self.robot.hand_slides.index(k)] = v
        self._add(np.repeat(self.last[None], n, 0), label, h0 + (h1 - h0) * w)

    def move(self, q_raw, dt_raw, label, carry=False):
        m = self.s.motion
        q_raw = np.asarray(q_raw, float)
        p = self.robot.fk_tcp(q_raw)[:, :3, 3]
        vj = np.abs(np.diff(q_raw, axis=0)).max() / dt_raw
        aj = np.abs(np.diff(q_raw, 2, axis=0)).max() / dt_raw ** 2 if len(q_raw) > 2 else 0.0
        vt = np.linalg.norm(np.diff(p, axis=0), axis=1).max() / dt_raw
        at = np.linalg.norm(np.diff(p, 2, axis=0), axis=1).max() / dt_raw ** 2 if len(p) > 2 else 0.0
        vmax, amax = (m.vmax_carry, m.amax_carry) if carry else (m.vmax_tool, m.amax_tool)
        k = max(1.0, vj / m.vmax_joint, math.sqrt(aj / m.amax_joint), vt / vmax, math.sqrt(at / amax))
        t = np.arange(len(q_raw)) * dt_raw * k
        spline = CubicSpline(t, q_raw, bc_type="clamped")
        self._add(spline(np.arange(self.dt, t[-1] + 1e-9, self.dt)), label)

    def path(self, q_rows, times, label):
        t = np.concatenate([[0.0], times])
        spline = CubicSpline(t, np.vstack([self.last[None], q_rows]), bc_type="clamped")
        self._add(spline(np.arange(self.dt, t[-1] + 1e-9, self.dt)), label)

    def stage(self, tube):
        self.stages.append((sum(len(a) for a in self.q), tube))

    def result(self):
        return Plan(np.concatenate(self.q), np.concatenate(self.hand), self.labels, self.stages)


class TaskPlanner:
    def __init__(self, settings, robot, cache: Path, address="127.0.0.1:50061"):
        self.s = settings
        self.robot = robot
        self.cache = Path(cache)
        self.address = address
        self.r_pick = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def tube_in_rack(self, rack, slot):
        x, y = self.s.rack.slot_xy(rack, slot)
        T = np.eye(4)
        T[:3, :3] = self.r_pick
        T[:3, 3] = (x, y, self.s.rack.floor + 0.5 * self.s.tube.length)
        return T

    def hand_rest(self):
        h = self.s.hand
        return np.array([h.bridge_cap, h.jaw_open, h.jaw_open, 0.0, 0.0, 0.0, 0.0])

    def load_or_plan(self):
        key = self.s.plan_key() + str(self.robot.urdf.stat().st_mtime)
        if self.cache.exists():
            f = np.load(self.cache)
            if str(f["key"]) == key:
                print(f"reusing the plan in {self.cache}")
                return Plan(f["q"], f["hand"], list(f["labels"]), [tuple(e) for e in f["stages"].tolist()])
        plan = self.plan()
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.cache, key=key, q=plan.q, hand=plan.hand, labels=np.array(plan.labels),
                 stages=np.array(plan.stages))
        return plan


    def _connect(self):
        channel = grpc.insecure_channel(self.address, options=[("grpc.max_receive_message_length", 64 << 20)])
        try:
            grpc.channel_ready_future(channel).result(timeout=5)
        except grpc.FutureTimeoutError:
            raise SystemExit(f"no cuRobo planner at {self.address}; start src/planner/planner_server.py first")
        self._channel = channel

    def _call(self, method, req):
        return json.loads(self._channel.unary_unary(f"/curobo.Planner/{method}")(json.dumps(req).encode(),
                                                                                timeout=300))

    @staticmethod
    def _box(name, center, dims):
        return {"name": name, "dims": list(dims), "pose": list(center) + [1.0, 0.0, 0.0, 0.0]}

    def plan(self):
        s, L, m, hs = self.s, self.s.layout, self.s.motion, self.s.hand
        self._connect()
        dt_raw = self._call("Info", {})["dt"]
        q_test = np.array(m.q_home) + 0.3
        err = np.abs(np.array(self._call("FK", {"q": [q_test.tolist()]})["poses"][0][:3])
                     - self.robot.fk_tcp(q_test)[0, :3, 3]).max()
        print(f"cuRobo planner: FK agreement {err * 1000:.3f} mm")

        top = s.rack.height
        beaker_h = s.beaker.height + 0.005
        fixed = [self._box("bench", (0.5, 0.0, -0.03), (1.6, 1.6, 0.04)),
                 self._box("rack_a", (L.rack_a[0], L.rack_a[1], top / 2), (s.rack.width, s.rack.length, top)),
                 self._box("rack_b", (L.rack_b[0], L.rack_b[1], top / 2), (s.rack.width, s.rack.length, top))]
        fixed += [self._box(f"beaker_{i}", (bx, by, beaker_h / 2), (2 * s.beaker.radius, 2 * s.beaker.radius, beaker_h))
                  for i, (bx, by) in enumerate(L.beakers)]
        n_tubes = L.n_tubes
        tl = Timeline(self.robot, s, m.q_home, self.hand_rest())
        self.tl = tl

        def scene_for(k):
            tubes = [(L.rack_a, sl) for sl in L.pick_slots[k + 1:n_tubes]] + [(L.rack_b, sl) for sl in L.place_slots[:k]]
            h = s.tube.cap_top + 0.005
            cuboids = fixed + [self._box(f"tube_{i}", (*s.rack.slot_xy(rack, sl), s.rack.floor + 0.5 * h), (0.03, 0.03, h))
                               for i, (rack, sl) in enumerate(tubes)]
            self._call("SetScene", {"cuboids": cuboids})

        def plan(xyz, R, mode, label, axis=None, fingers=False, carry=False):
            req = {"start": tl.last.tolist(), "pose": list(map(float, xyz)) + quat_wxyz(R), "mode": mode,
                   "ignore_fingers": fingers}
            if axis:
                req["axis"] = axis
            r = self._call("PlanPose", req)
            if not r["success"]:
                raise SystemExit(f"cuRobo could not plan: {label}")
            tl.move(r["q"], dt_raw, label, carry)

        def ik_rows(poses, seeds=None):
            req = {"start": tl.last.tolist(), "ignore_fingers": True,
                   "poses": [list(map(float, p)) + quat_wxyz(Rm) for p, Rm in poses]}
            if seeds is not None:
                req["seeds"] = [list(map(float, q)) for q in seeds]
            r = self._call("IKPath", req)
            if not r["success"]:
                raise NoRoom()
            q = np.array(r["q"])
            if np.abs(np.diff(np.vstack([tl.last[None], q]), axis=0)).max() > math.radians(20.0):
                raise NoRoom()
            return q

        def home_like(p, Rm):
            r = self._call("IKPath", {"start": list(m.q_home), "ignore_fingers": True,
                                      "poses": [list(map(float, p)) + quat_wxyz(Rm)]})
            return np.array(r["q"][0]) if r["success"] else None

        def carry_line(p_to, R_to, label):
            T0 = self.robot.fk_tcp(tl.last)[0]
            p_from, R_from = T0[:3, 3], T0[:3, :3]
            rel = R_to @ R_from.T
            turn = math.atan2(rel[1, 0], rel[0, 0])
            r0, r1 = math.hypot(*p_from[:2]), math.hypot(*p_to[:2])
            a0, a1 = math.atan2(p_from[1], p_from[0]), math.atan2(p_to[1], p_to[0])
            n = max(8, int(max(abs(a1 - a0) * max(r0, r1), abs(r1 - r0), abs(p_to[2] - p_from[2])) / 0.02))
            poses = []
            for u in np.linspace(1.0 / n, 1.0, n):
                w = smooth(u)
                r, az = r0 + w * (r1 - r0), a0 + w * (a1 - a0)
                poses.append((np.array([r * math.cos(az), r * math.sin(az), p_from[2] + w * (p_to[2] - p_from[2])]),
                              rot_axis([0.0, 0.0, 1.0], w * turn) @ R_from))
            q_end = home_like(*poses[-1])
            seeds = None if q_end is None else [tl.last + smooth(u) * (q_end - tl.last) for u in np.linspace(1.0 / n, 1.0, n)]
            tl.move(np.vstack([tl.last[None], ik_rows(poses, seeds)]), 0.05, label, carry=True)

        lip = s.lip_height()

        def bow(lip0, R0, a, angles, label):
            t = np.cross([0.0, 0.0, 1.0], a)
            idx = list(range(m.ik_every, len(angles), m.ik_every))
            if not idx or idx[-1] != len(angles) - 1:
                idx.append(len(angles) - 1)
            poses = []
            for i in idx:
                w = smooth((angles[i] - 30.0) / 45.0)
                lip_p = lip0 + m.lip_in * w * a - [0.0, 0.0, m.lip_drop * w]
                R = rot_axis(t, math.radians(angles[i])) @ R0
                poses.append((lip_p - lip * (R @ [0.0, 1.0, 0.0]), R))
            tl.path(ik_rows(poses), np.array(idx) * tl.dt, label)

        def pour_at(k, b, R, carry_z):
            name = f"tube {k + 1}"
            bx, by = L.beakers[b]
            a = -np.array([bx, by, 0.0]) / math.hypot(bx, by)
            R_st = rot_axis([0.0, 0.0, 1.0], math.atan2(by, bx)) @ R
            lip0 = np.array([bx, by, s.beaker.height + m.lip_start]) - m.front_d * a
            front = lip0 - [0.0, 0.0, lip]
            carry_line(front + [0.0, 0.0, carry_z - front[2]], R_st, f"{name}: carry to beaker {b + 1}")
            plan(front, R_st, "linear", f"{name}: down beside the beaker", axis="y", fingers=True, carry=True)
            R0 = self.robot.fk_tcp(tl.last)[0, :3, :3]
            liq = L.liquids[b]
            bow(lip0, R0, a, self.roll_profile(0.0, m.tilt_max, m.tilt_fast, liq.tilt_slow, m.pour_from, m.pour_to),
                f"{name}: pour")
            tl.hold(liq.drain, f"{name}: drain")
            bow(lip0, R0, a, self.roll_profile(m.tilt_max, 0.0, m.tilt_fast * 1.5), f"{name}: straighten up")
            plan(front + [0.0, 0.0, carry_z - front[2]], R_st, "linear", f"{name}: up", axis="y", fingers=True,
                 carry=True)

        def twist(strokes, sign, label):
            for _ in range(strokes):
                tl.hand_to(0.7, label, up_left=tl.h("up_left") + sign * hs.twist_stroke,
                           up_right=tl.h("up_right") - sign * hs.twist_stroke)
                tl.hand_to(0.25, label, up_jaw=hs.jaw_ease)
                tl.hand_to(0.5, label, up_left=tl.h("up_left") - sign * hs.twist_stroke,
                           up_right=tl.h("up_right") + sign * hs.twist_stroke)
                tl.hand_to(0.25, label, up_jaw=hs.jaw_cap)

        def decap(name):
            tl.hand_to(0.5, f"{name}: fingertips to the cap", up_left=hs.finger_mid, up_right=hs.finger_mid)
            tl.hand_to(0.5, f"{name}: grip the cap", up_jaw=hs.jaw_cap)
            twist(hs.twist_strokes, 1.0, f"{name}: twist the cap loose")
            tl.hand_to(0.8, f"{name}: lift the cap clear", bridge=hs.bridge_cap + hs.cap_clear)
            tl.hand_to(0.8, f"{name}: tuck the cap in", up_left=0.0, up_right=0.0)
            tl.hand_to(0.6, f"{name}: tuck the cap in", bridge=hs.bridge_cap)

        def recap(name):
            tl.hand_to(0.6, f"{name}: cap up", bridge=hs.bridge_cap + hs.cap_clear)
            tl.hand_to(0.8, f"{name}: cap back over the tube", up_left=hs.finger_mid, up_right=hs.finger_mid)
            tl.hand_to(0.8, f"{name}: cap down", bridge=hs.bridge_cap)
            twist(hs.twist_strokes, -1.0, f"{name}: twist the cap shut")
            tl.hand_to(0.4, f"{name}: let go of the cap", up_jaw=hs.jaw_open)
            tl.hand_to(0.4, f"{name}: fingertips home", up_left=0.0, up_right=0.0)

        grasp_z = s.tube.grasp_y - 0.5 * s.tube.length
        up = np.array([0.0, 0.0, m.lift])
        above = np.array([0.0, 0.0, m.approach])
        R = self.r_pick

        def plan_tube(k):
            name = f"tube {k + 1}"
            grasp = self.tube_in_rack(L.rack_a, L.pick_slots[k])[:3, 3] + [0.0, 0.0, grasp_z]
            place = self.tube_in_rack(L.rack_b, L.place_slots[k])[:3, 3] + [0.0, 0.0, grasp_z]
            if k:
                tl.stage(k)
            scene_for(k)
            plan(grasp + above, R, "free", f"{name}: over the tube")
            plan(grasp, R, "linear", f"{name}: down over the tube", axis="y", fingers=True)
            tl.hand_to(1.0, f"{name}: grasp", down_jaw=hs.jaw_tube)
            plan(grasp + up, R, "linear", f"{name}: lift", axis="y", fingers=True, carry=True)
            decap(name)
            pour_at(k, L.tube_beaker[k], R, grasp[2] + m.lift)
            recap(name)
            carry_line(place + up, R, f"{name}: carry to rack B")
            plan(place, R, "linear", f"{name}: insert", axis="y", fingers=True, carry=True)
            tl.hand_to(0.8, f"{name}: release", down_jaw=hs.jaw_open)
            plan(place + above, R, "linear", f"{name}: up off the tube", axis="y", fingers=True)

        tl.stage(0)
        tl.hold(1.5, "ready")
        for k in range(n_tubes):
            try:
                plan_tube(k)
            except NoRoom:
                raise SystemExit(f"tube {k + 1}: cuRobo found no smooth IK path (carry or pour)")
            print(f"  planned tube {k + 1}/{n_tubes}", flush=True)
        tl.stage(n_tubes)
        scene_for(n_tubes)
        r = self._call("PlanJoints", {"start": tl.last.tolist(), "goal": list(m.q_home)})
        if not r["success"]:
            raise SystemExit("cuRobo could not plan: home")
        tl.move(r["q"], dt_raw, "home")
        tl.hold(3.0, "done")
        return tl.result()

    def roll_profile(self, a0, a1, fast, slow=None, slow_from=0.0, slow_to=0.0):
        dt = self.s.fluid.dt_sub
        sgn = 1.0 if a1 >= a0 else -1.0
        out, a, t = [a0], a0, 0.0
        while sgn * (a1 - a) > 0.01:
            speed = fast
            if slow is not None:
                w = smooth((a - slow_from + 10.0) / 10.0) * (1.0 - smooth((a - slow_to) / 10.0))
                speed = fast + (slow - fast) * w
            speed *= max(0.05, smooth(t / 0.8)) * max(0.05, smooth(abs(a1 - a) / 8.0))
            a += sgn * speed * dt
            t += dt
            out.append(min(a, a1) if sgn > 0 else max(a, a1))
        return np.array(out)
