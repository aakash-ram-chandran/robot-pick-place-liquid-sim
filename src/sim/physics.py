import gc
import math
import struct

import numpy as np
import threepp as tp

from geometry import body_pose, quat_wxyz, rot_axis, rpy_matrix, set_pose, to_world


def load_stl(path):
    d = path.read_bytes()
    if d[:5] == b"solid" and b"facet" in d[:300]:
        return np.array([[float(v) for v in l.split()[1:4]] for l in d.decode().splitlines()
                         if l.strip().startswith("vertex")])
    n = struct.unpack("<I", d[80:84])[0]
    rows = np.frombuffer(d[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rows["v"].reshape(-1, 3).astype(float)


def thin(p, n=400):
    p = np.unique(np.round(p, 5), axis=0)
    return p[::max(1, len(p) // n)]


def bbox_corners(p):
    lo, hi = p.min(0), p.max(0)
    return np.array([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


class PhysicsWorld:
    dofs = [f"link_{i}" for i in range(1, 8)] + ["bridge"] + [f"{s}_{l}{t}" for s in ("left", "right")
                                                              for l in ("down", "up") for t in ("", "_tip")]
    hand_links = ["link_7", "bridge", "left_down", "right_down", "left_up", "right_up",
                  "left_down_tip", "right_down_tip", "left_up_tip", "right_up_tip"]

    def __init__(self, settings, robot, plan, tube_points, bench, rack_colliders, tube_poses, adapter):
        self.s = settings
        self.robot = robot
        self.plan = plan
        self.tube_pts, self.cap_pts, self.capped_pts = tube_points
        self.bench = bench
        self.rack_colliders = rack_colliders
        self.tube_poses = tube_poses
        self.adapter = adapter
        T0 = robot.fk(np.zeros(7))
        self.T_build = T0
        self.T7h = np.linalg.inv(T0["link_7"][0]) @ T0["hand"][0]
        self.world = None
        self.active = -1
        self.poses = []


    @staticmethod
    def _collider(points, T):
        m = tp.Mesh(tp.ConvexGeometry([tp.Vector3(*map(float, q)) for q in points]), tp.MeshBasicMaterial())
        set_pose(m, T)
        return m

    def _hand_points(self, name):
        path = self.robot.hand_meshes / f"{name}.obj"
        return np.array([[float(v) for v in l.split()[1:4]] for l in open(path) if l.startswith("v ")]) \
            * (0.001 * self.s.hand.scale)

    def _build_robot(self):
        ph, hs, T0 = self.s.physics, self.s.hand, self.T_build
        art = self.world.create_articulation(fixed_base=True, solver_position_iterations=32,
                                             disable_self_collision=True)
        pad = self.world.create_material(hs.pad_friction, 0.9 * hs.pad_friction, 0.0, "average")
        links = {"link_0": art.add_link(self._collider(thin(load_stl(self.robot.mesh_path("link_0"))),
                                                       to_world(T0["link_0"][0])), density=1500.0)}
        for i in range(1, 8):
            name = f"link_{i}"
            T = to_world(T0[name][0])
            pts = thin(load_stl(self.robot.mesh_path(name)))
            if i == 7:
                a = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
                post = [(self.adapter.clamp_r * math.cos(t), self.adapter.clamp_r * math.sin(t), z)
                        for t in a for z in (self.adapter.flange_z, self.adapter.post_end)]
                base = thin(self._hand_points("base__nylon")) @ self.T7h[:3, :3].T + self.T7h[:3, 3]
                pts = np.vstack([pts, post, base])
            lim = math.radians(ph.arm_limits_deg[i - 1])
            links[name] = art.add_link(self._collider(pts, T), parent=links[f"link_{i - 1}"], density=1500.0,
                                       axis=list(map(float, T[:3, 2])), anchor=list(map(float, T[:3, 3])),
                                       lower=-lim, upper=lim, stiffness=ph.arm_stiffness, damping=ph.arm_damping,
                                       max_force=ph.arm_torque[i - 1])
        TH = to_world(T0["hand"][0])

        def slide(name, parent, pts, axis, upper, force, material=None):
            links[name] = art.add_link(self._collider(pts, TH), parent=links[parent], density=1200.0,
                                       axis=list(map(float, axis)), anchor=list(map(float, TH[:3, 3])), lower=0.0,
                                       upper=upper, stiffness=hs.slide_stiffness, damping=hs.slide_damping,
                                       max_force=force, joint_type="prismatic", material=material)

        slide("bridge", "link_7", thin(self._hand_points("bridge__nylon")), TH[:3, 2], 0.052, 30.0)
        for side, sgn in (("left", -1.0), ("right", 1.0)):
            for lvl, parent in (("down", "link_7"), ("up", "bridge")):
                slide(f"{side}_{lvl}", parent, bbox_corners(self._hand_points(f"{side}_{lvl}_rack__steel")),
                      sgn * TH[:3, 1], 0.052, hs.grip_force)
                slide(f"{side}_{lvl}_tip", f"{side}_{lvl}",
                      bbox_corners(self._hand_points(f"{side}_{lvl}_finger__nylon")), TH[:3, 0], 0.065,
                      hs.grip_force, pad)
        art.finalize()
        return art, links

    def _rigid(self, points, T, density=1200.0):
        g = tp.BufferGeometry()
        g.set_attribute("position", points.astype(np.float32))
        m = tp.Mesh(g, tp.MeshBasicMaterial())
        set_pose(m, T)
        return self.world.add_dynamic_convex(m, density)

    def _stopper(self, tube, cap, T, pull):
        R = T[:3, :3] @ np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        w, qx, qy, qz = quat_wxyz(R)
        prm = tp.Joint.Params()
        prm.type = tp.Joint.Type.REVOLUTE
        prm.damping = 0.002
        prm.max_force = self.s.physics.cap_friction
        prm.break_force = pull
        lip = T[:3, :3] @ [0.0, self.s.tube.y_top, 0.0] + T[:3, 3]
        return tp.Joint(self.world, tube, cap, tp.Vector3(*lip), tp.Quaternion(qx, qy, qz, w), prm)

    def build(self):
        self.world = self.art = self.links = None
        self.tubes, self.caps, self.seals = [], [], []
        gc.collect()
        self.world = tp.PhysxWorld(fixed_timestep=1.0 / self.s.physics.hz, max_substeps=64, tgs_pcm=True)
        self.world.add_static(self.bench)
        for m in self.rack_colliders:
            self.world.add_static_trimesh(m)
        self.art, self.links = self._build_robot()
        self.art.set_joint_positions(self.targets(0))
        self.art.set_drive_targets(self.targets(0))
        rng = np.random.default_rng(7)
        n = self.s.layout.n_tubes
        for k in range(n):
            T = self.tube_poses[k].copy()
            T[1, 3] += 0.002
            lean = rng.uniform(-0.015, 0.015, 2)
            T[:3, :3] = rpy_matrix(lean[0], 0.0, lean[1]) @ T[:3, :3]
            self.tubes.append(self._rigid(self.capped_pts, T))
            spin = rng.uniform(-0.6, 0.6, 2)
            self.tubes[k].set_angular_velocity(tp.Vector3(spin[0], 0.0, spin[1]))
        self.caps = [None] * n
        self.seals = [None] * n
        for _ in range(180):
            self.world.step(self.s.fluid.dt)
        self.t0 = self.world.sim_time
        self.world.on_pre_substep(self._drive)
        self.world.on_post_substep(self._record)


    def targets(self, i):
        hs, p = self.s.hand, self.plan
        bridge, dj, uj, dl, dr, ul, ur = p.hand[i]
        label = p.labels[i]
        if label.endswith(("cap down", "twist the cap shut")):
            bridge -= hs.press
        dj_right = p.hand[max(i - int(hs.release_lag / self.s.fluid.dt_sub), 0)][1] if label.endswith(": release") else dj
        if dj < hs.jaw_open - 1e-4:
            dj -= hs.squeeze
        if dj_right < hs.jaw_open - 1e-4:
            dj_right -= hs.squeeze
        if uj < hs.jaw_ease - 1e-4:
            uj -= hs.squeeze
        return np.array([*p.q[i], bridge, dj, dl, uj, ul, dj_right, dr, uj, ur], np.float32)

    def substep_index(self):
        return min(int(round((self.world.sim_time - self.t0) / self.s.fluid.dt_sub)), len(self.plan) - 2)

    def _drive(self, dt):
        i = self.substep_index()
        self.art.set_drive_targets(self.targets(i))
        for name, w in zip(self.dofs, (self.plan.q[i + 1] - self.plan.q[i]) / self.s.fluid.dt_sub):
            self.links[name].set_drive_velocity(float(w))

    def _record(self, dt):
        if self.active >= 0:
            self.poses.append((self.world.sim_time, body_pose(self.tubes[self.active])))

    def step(self, dt, active):
        self.active = active
        self.poses = []
        self.world.step(dt)
        return self.poses


    def _replace(self, old, parts, T):
        lin, ang = old[0].linear_velocity, old[0].angular_velocity
        for b in old:
            self.world.remove(b)
        new = [self._rigid(pts, T) for pts in parts]
        for b in new:
            b.set_linear_velocity(lin)
            b.set_angular_velocity(ang)
        return new

    def split(self, k):
        T = body_pose(self.tubes[k])
        self.tubes[k], self.caps[k] = self._replace([self.tubes[k]], [self.tube_pts, self.cap_pts], T)
        self.seals[k] = self._stopper(self.tubes[k], self.caps[k], T, self.s.physics.seal_pull)

    def merge(self, k):
        if self.caps[k] is None or self.seals[k] is None or self.seals[k].broken:
            return
        T = body_pose(self.tubes[k])
        self.seals[k] = None
        (self.tubes[k],) = self._replace([self.tubes[k], self.caps[k]], [self.capped_pts], T)
        self.caps[k] = None

    def seat_cap(self, k):
        if self.seals[k] is None or not self.seals[k].broken:
            return
        Tt = body_pose(self.tubes[k])
        rel = np.linalg.inv(Tt) @ body_pose(self.caps[k])
        if math.hypot(rel[0, 3], rel[2, 3]) < 0.0025 and abs(rel[1, 3]) < 0.0015 and rel[1, 1] > math.cos(math.radians(6)):
            on = np.eye(4)
            on[:3, :3] = rot_axis([0.0, 1.0, 0.0], math.atan2(rel[0, 2], rel[0, 0]))
            C = Tt @ on
            w, qx, qy, qz = quat_wxyz(C[:3, :3])
            self.caps[k].set_pose(tp.Vector3(*C[:3, 3]), tp.Quaternion(qx, qy, qz, w))
            self.seals[k] = self._stopper(self.tubes[k], self.caps[k], Tt, self.s.physics.seat_pull)

    def cap_off(self, k):
        return self.seals[k] is not None and self.seals[k].broken

    def link_pose(self, name):
        return body_pose(self.links[name])

    def tube_pose(self, k):
        return body_pose(self.tubes[k])

    def cap_pose(self, k):
        return body_pose(self.caps[k]) if self.caps[k] is not None else body_pose(self.tubes[k])
