from pathlib import Path

import numpy as np
import threepp as tp

from fluid import Fluid, Surface
from geometry import blend, set_pose, to_world
from physics import PhysicsWorld
from planning import TaskPlanner
from robot import RobotModel
from scene import Adapter, Beaker, Materials, Rack, RobotVisuals, Tube, studio_env


class Demo:
    FAR = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, -10.0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])

    def __init__(self, settings, assets: Path, cache: Path, width=1400, height=860, renderer="gl",
                 headless=False, title="robot liquid handling"):
        self.s = s = settings
        self.n = s.layout.n_tubes
        self.robot = RobotModel(Path(assets) / "iiwa14_cartesian_hand.urdf",
                                Path(assets) / "cartesian_hand" / "meshes")
        self.planner = TaskPlanner(s, self.robot, cache)
        self.plan = self.planner.load_or_plan()
        self.n_frames = len(self.plan) // s.fluid.substeps
        print(f"task: {len(self.plan) * s.fluid.dt_sub:.0f} s for {self.n} tubes")

        self.vulkan = renderer == "vulkan"
        if self.vulkan:
            self.canvas = tp.Canvas(title, width=width, height=height, vsync=False, headless=headless)
            self.renderer = tp.VulkanRenderer(self.canvas)
        else:
            self.canvas = tp.Canvas(title, width=width, height=height, antialiasing=4, headless=headless)
            self.renderer = tp.GLRenderer(self.canvas)
            self.renderer.shadow_map_enabled = True
        self.renderer.tone_mapping = tp.ToneMapping.ACESFilmic
        self.renderer.tone_mapping_exposure = 1.0

        self.scene = tp.Scene()
        self.scene.environment = studio_env(2048, 1024, dark=True) if self.vulkan else studio_env()
        self.scene.background = tp.Color(0x1c1f23)
        self.look = (0.45, 0.1, 0.0)
        self.camera = tp.PerspectiveCamera(38, self.canvas.aspect(), 0.005, 20.0)
        self.camera.position.set(0.95, 0.62, 0.85)
        self.camera.look_at(*self.look)
        self._lights()
        self._build_scene()

        self.fluid = Fluid(s)
        self._build_liquid()
        self.physics = PhysicsWorld(s, self.robot, self.plan, self.tube.collider_points(), self.bench,
                                    self.rack_colliders, self.tubes_upright, self.adapter)
        self.reset()
        self.place()
        for surf, m in zip((self.surf_tube, self.surf_world), self.live_meshes):
            surf.geometry.set_draw_range(0, 3)
            m.visible = True
        self.renderer.render(self.scene, self.camera)
        for surf, m in zip((self.surf_tube, self.surf_world), self.live_meshes):
            surf.register(self.renderer, m)
        self._hide_live()


    def _lights(self):
        target = tp.Object3D()
        target.position.set(*self.look)
        self.scene.add(target)
        sun = tp.DirectionalLight(tp.Color(0xfff6ea), 2.4)
        sun.position.set(self.look[0] + 1.0, self.look[1] + 2.0, self.look[2] + 1.2)
        sun.set_target(target)
        sun.cast_shadow = True
        sun.set_shadow_frustum(-0.7, 0.7, 0.7, -0.7)
        sun.set_shadow_bias(-0.0003)
        self.scene.add(sun)
        self.scene.add(tp.HemisphereLight(tp.Color(0xdde6f0), tp.Color(0x40444a), 0.6))

    def _build_scene(self):
        s, L = self.s, self.s.layout
        self.bench = tp.Mesh(tp.BoxGeometry(1.8, 0.04, 1.4), Materials.standard(0x3a3f45, 0.7))
        self.bench.position.set(0.45, -0.02, 0.0)
        self.bench.receive_shadow = True
        self.scene.add(self.bench)
        rack = Rack(s.rack)
        self.rack_colliders = []
        for cx, cy in (L.rack_a, L.rack_b):
            self.rack_colliders += rack.build(self.scene, (cx, 0.0, -cy))
        beaker = Beaker(s.beaker)
        for bx, by in L.beakers:
            m = tp.Mesh(beaker.geometry(), Materials.glass(0.002, thin_walled=True))
            m.position.set(bx, 0.0, -by)
            self.scene.add(m)

        visuals = RobotVisuals(self.robot, s.hand.scale)
        self.link_objs = {}
        for link in self.robot.link_order:
            if link.startswith("link_"):
                self.link_objs[link] = visuals.link(link)
                self.scene.add(self.link_objs[link])
        self.adapter = Adapter()
        self.link_objs["link_7"].add(self.adapter.group())
        self.hand_objs = []
        for files, _ in self.robot.hand_parts:
            self.hand_objs.append(visuals.hand_part(files))
            self.scene.add(self.hand_objs[-1])

        self.tube = Tube(s.tube)
        self.tube_objs, self.cap_objs = [], []
        for _ in range(self.n):
            t, c = self.tube.meshes()
            self.scene.add(t)
            self.scene.add(c)
            self.tube_objs.append(t)
            self.cap_objs.append(c)
        self.tubes_upright = [to_world(self.planner.tube_in_rack(L.rack_a, sl)) for sl in L.pick_slots[:self.n]]

    def _build_liquid(self):
        s, t, cell = self.s, self.s.tube, self.fluid.cell
        wdim = (int(0.090 / cell) + 1, int(0.120 / cell) + 1, int(0.090 / cell) + 1)
        self.tg0 = (-(t.r_out + 0.0015), t.y_cap - t.r_out - 0.002, -(t.r_out + 0.0015))
        tdim = (int(-2 * self.tg0[0] / cell) + 1, int((t.y_top + 0.003 - self.tg0[1]) / cell) + 1,
                int(-2 * self.tg0[0] / cell) + 1)
        self.surf_tube = Surface(self.fluid, tdim, s.fluid.max_tris_tube, 1)
        self.surf_world = Surface(self.fluid, wdim, s.fluid.max_tris, 0)
        self.live_mat = Materials.liquid(s.layout.liquids[0].color)
        self.live_meshes = []
        for surf in (self.surf_tube, self.surf_world):
            m = tp.Mesh(surf.geometry, self.live_mat)
            m.cast_shadow = True
            m.frustum_culled = False
            self.scene.add(m)
            self.live_meshes.append(m)
        still = self.surf_tube.bake(self.fluid.fill_tube(np.eye(4)), self.tg0, np.eye(4))
        self.still = []
        for k in range(self.n):
            m = tp.Mesh(still, Materials.liquid(s.layout.liquids[s.layout.tube_beaker[k]].color))
            m.cast_shadow = True
            self.scene.add(m)
            self.still.append(m)
        self.beaker_liquid = []
        for b in range(len(s.layout.beakers)):
            m = tp.Mesh(tp.BufferGeometry(), Materials.liquid(s.layout.liquids[b].color))
            self.scene.add(m)
            self.beaker_liquid.append(m)
        mat = tp.MeshPhysicalMaterial()
        mat.color = 0xffffff
        mat.roughness = 0.05
        mat.transparent = True
        mat.opacity = 0.6
        mat.clearcoat = 1.0
        mat.depth_test = False
        n_b = max(1, max(q.bubbles for q in s.layout.liquids))
        self.bubbles = tp.InstancedMesh(tp.SphereGeometry(s.fluid.bubble_r, 8, 6), mat, n_b)
        self.bubbles.frustum_culled = False
        self.bubbles.render_order = 10
        self.bubbles.set_count(0)
        self.scene.add(self.bubbles)


    def reset(self):
        self.frame = 0
        self.active = -1
        self.beaker_count = [0] * len(self.s.layout.beakers)
        self.push = np.zeros(3)
        self.split_ml = (0.0, 0.0)
        self.physics.build()
        for m in self.still:
            m.visible = True
        for m in self.beaker_liquid:
            m.set_geometry(tp.BufferGeometry())
        self._hide_live()

    def _hide_live(self):
        self.surf_tube.hide()
        self.surf_world.hide()

    def beaker_center(self, b):
        bx, by = self.s.layout.beakers[b]
        return np.array([bx, 0.0, -by])

    def beaker_origin(self, b):
        c = self.beaker_center(b)
        return (c[0] - 0.045, -0.002, c[2] - 0.045)

    def _freeze_live(self):
        k = self.active
        if k < 0:
            return
        b = self.s.layout.tube_beaker[k]
        _, _, in_b = self.fluid.split_ml(self.physics.tube_pose(k), self.beaker_center(b), 0)
        self.beaker_count[b] += in_b
        pts = self.fluid.fill_beaker(self.beaker_center(b), self.beaker_count[b])
        self.beaker_liquid[b].set_geometry(self.surf_world.bake(pts, self.beaker_origin(b), self.FAR))
        self.active = -1
        self._hide_live()

    def _start_tube(self, k):
        self._freeze_live()
        if k >= self.n:
            return
        self.active = k
        b = self.s.layout.tube_beaker[k]
        c = self.beaker_center(b)
        count = self.beaker_count[b]
        floor = self.s.beaker.floor if count == 0 else float(self.fluid.fill_beaker(c, count)[:, 1].max()) + 0.5 * self.fluid.d
        self.fluid.start(self.fluid.fill_tube(self.physics.tube_pose(k)), self.s.layout.liquids[b], (c[0], c[2]), floor)
        self.still[k].visible = False
        self.live_mat.color = self.s.layout.liquids[b].color

    def step(self):
        s, p, ph = self.s, self.plan, self.physics
        sub, dt = s.fluid.substeps, s.fluid.dt
        i0, i1 = self.frame * sub, min((self.frame + 1) * sub, len(p) - 1)
        for idx, k in p.stages:
            if i0 < idx <= i1 or (idx == 0 and self.frame == 0):
                self._start_tube(k)
        k, label = self.active, p.labels[i1]
        if 0 <= k < self.n:
            if ph.caps[k] is None and ph.seals[k] is None and label.endswith(": fingertips to the cap"):
                ph.split(k)
            elif ph.caps[k] is not None and label.endswith(": insert"):
                ph.merge(k)
        start = None
        if k >= 0:
            ph.tubes[k].add_impulse(tp.Vector3(*(self.push * dt)))
            start = ph.tube_pose(k)
        t_start = ph.world.sim_time
        poses = ph.step(dt, k)
        if k >= 0 and label.endswith(("cap down", "twist the cap shut")):
            ph.seat_cap(k)
        self.frame += 1
        if k < 0:
            self._step_bubbles()
            return
        times = np.array([t_start] + [t for t, _ in poses])
        Ts = [start] + [T for _, T in poses]
        subs = []
        for n in range(1, sub + 1):
            t = t_start + n * (times[-1] - t_start) / sub
            j = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1))
            subs.append(blend(Ts[j - 1], Ts[j], (t - times[j - 1]) / max(times[j] - times[j - 1], 1e-9)))
        p0, mass = self.fluid.momentum_in_tube(start, 1)
        self.fluid.step(np.array(subs))
        p1, _ = self.fluid.momentum_in_tube(start, 0)
        self.push = mass * np.array([0.0, -9.81, 0.0]) - (p1 - p0) / dt
        self._update_live(subs[-1])
        self._step_bubbles()

    def _update_live(self, T):
        b = self.s.layout.tube_beaker[self.active]
        self.surf_tube.update(self.tg0, T)
        self.surf_world.update(self.beaker_origin(b), T)

    def _step_bubbles(self):
        L, k = self.s.layout, self.active
        seed = self.frame * 7919
        if k >= 0 and L.liquids[L.tube_beaker[k]].bubbles > 0:
            n = L.liquids[L.tube_beaker[k]].bubbles
            pts = self.fluid.step_bubbles(n, seed)
        else:
            settled = [b for b, q in enumerate(L.liquids) if q.bubbles > 0 and self.beaker_count[b] > 0]
            if not settled:
                self.bubbles.set_count(0)
                return
            b = settled[0]
            n = L.liquids[b].bubbles
            c = self.beaker_center(b)
            top = float(self.fluid.fill_beaker(c, self.beaker_count[b])[:, 1].max())
            pts = self.fluid.step_bubbles(n, seed, (c[0], c[2], self.s.beaker.floor), top)
        m = tp.Matrix4()
        for i, q in enumerate(pts):
            m.make_translation(float(q[0]), float(q[1]), float(q[2]))
            self.bubbles.set_matrix_at(i, m)
        self.bubbles.set_count(n)
        self.bubbles.instance_matrix_needs_update()

    def place(self):
        ph = self.physics
        for name, obj in self.link_objs.items():
            set_pose(obj, ph.link_pose(name))
        for obj, name in zip(self.hand_objs, ph.hand_links):
            T = ph.link_pose(name)
            set_pose(obj, T @ ph.T7h if name == "link_7" else T)
        for k in range(self.n):
            T = ph.tube_pose(k)
            set_pose(self.tube_objs[k], T)
            set_pose(self.still[k], T)
            set_pose(self.cap_objs[k], ph.cap_pose(k))

    def liquid_split(self):
        k = self.active
        if k < 0:
            return 0.0, 0.0
        b = self.s.layout.tube_beaker[k]
        tube_ml, beaker_ml, _ = self.fluid.split_ml(self.physics.tube_pose(k), self.beaker_center(b),
                                                    self.beaker_count[b])
        return tube_ml, beaker_ml

    def label(self):
        return self.plan.labels[min(self.frame * self.s.fluid.substeps, len(self.plan) - 1)]

    def render(self):
        self.renderer.render(self.scene, self.camera)

    def save_frame(self, path):
        if self.vulkan:
            self.renderer.save_frame(self.scene, self.camera, str(path))
        else:
            self.renderer.render(self.scene, self.camera)
            self.renderer.save_frame(str(path))
