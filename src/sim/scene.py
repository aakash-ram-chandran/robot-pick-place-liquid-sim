import math

import numpy as np
import threepp as tp

from geometry import pose, rot_axis, set_pose


class Materials:
    @staticmethod
    def standard(color, roughness=0.6, metalness=0.0):
        m = tp.MeshStandardMaterial()
        m.color = color
        m.roughness = roughness
        m.metalness = metalness
        return m

    @staticmethod
    def glass(thickness, ior=1.5, roughness=0.02, thin_walled=False):
        m = tp.MeshPhysicalMaterial()
        m.color = 0xffffff
        m.metalness = 0.0
        m.roughness = roughness
        m.transmission = 1.0
        m.ior = ior
        m.thickness = thickness
        m.specular_intensity = 1.0
        m.thin_walled = thin_walled
        return m

    @staticmethod
    def liquid(color):
        m = tp.MeshPhysicalMaterial()
        m.color = color
        m.metalness = 0.0
        m.roughness = 0.12
        m.clearcoat = 1.0
        m.clearcoat_roughness = 0.03
        m.side = tp.Side.Double
        return m

    @staticmethod
    def plastic(color, roughness=0.45):
        m = tp.MeshPhysicalMaterial()
        m.color = color
        m.roughness = roughness
        m.metalness = 0.0
        m.clearcoat = 0.3
        m.clearcoat_roughness = 0.4
        return m


def studio_env(w=512, h=256, dark=False):
    elev = ((np.arange(h, dtype=np.float32) + 0.5) / h - 0.5) * math.pi
    az = ((np.arange(w, dtype=np.float32) + 0.5) / w - 0.5) * 2.0 * math.pi
    el, az = np.meshgrid(elev, az, indexing="ij")
    y = np.sin(el)
    if dark:
        col = np.where(y >= 0.0, 0.030 + 0.030 * y, 0.022)[..., None] * np.float32([1.0, 1.0, 1.04])
    else:
        col = np.where(y >= 0.0, 0.16 + 0.20 * y, 0.10 + 0.04 * y)[..., None] * np.float32([1.0, 1.0, 1.03])
    edge = 80.0 if dark else 12.0

    def softbox(az0, el0, daz, delv, power):
        daz_ = np.angle(np.exp(1j * (az - math.radians(az0))))
        u = np.abs(daz_) / math.radians(daz)
        t = np.abs(el - math.radians(el0)) / math.radians(delv)
        return power / (1.0 + np.exp(np.minimum(edge * (np.maximum(u, t) - 1.0), 60.0)))

    col = col + (softbox(45, 20, 14, 32, 7.0) + softbox(135, 25, 18, 30, 3.0)
                 + softbox(-90, 30, 30, 18, 2.0))[..., None] * np.float32([1.0, 0.98, 0.95])
    top = (y > 0.9).astype(np.float32) if dark else np.clip((y - 0.85) / 0.15, 0.0, 1.0)
    col = col + (top * 2.5)[..., None]
    out = np.ones((h, w, 4), np.float32)
    out[..., :3] = col
    return tp.float_texture(out)


def lathe(profile, segments=96):
    prof = np.asarray(profile, np.float32)
    th = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    c, s = np.cos(th)[None, :], np.sin(th)[None, :]
    r, y, nr, ny = (prof[:, i:i + 1] for i in range(4))
    pos = np.stack([r * c, np.broadcast_to(y, (len(prof), segments)), r * s], axis=-1)
    nrm = np.stack([nr * c, np.broadcast_to(ny, (len(prof), segments)), nr * s], axis=-1)
    i = np.arange(len(prof) - 1)[:, None]
    j = np.arange(segments)[None, :]
    a, b = i * segments + j, (i + 1) * segments + j
    cc, d = i * segments + (j + 1) % segments, (i + 1) * segments + (j + 1) % segments
    idx = np.concatenate([np.stack([a, b, cc], -1).reshape(-1, 3), np.stack([b, d, cc], -1).reshape(-1, 3)])
    g = tp.BufferGeometry()
    g.set_attribute("position", pos.reshape(-1, 3).astype(np.float32))
    g.set_attribute("normal", nrm.reshape(-1, 3).astype(np.float32))
    g.set_index(idx.astype(np.uint32))
    return g


def rounded_rim(r_in, r_out, y, n=10):
    w, rm = 0.5 * (r_out - r_in), 0.5 * (r_in + r_out)
    return [(rm + w * math.cos(b), y + w * math.sin(b), math.cos(b), math.sin(b)) for b in np.linspace(0.0, math.pi, n)]


def rounded_rect_shape(w, d, r):
    s = tp.Shape()
    x, y = w / 2, d / 2
    s.move_to(-x + r, -y)
    s.line_to(x - r, -y)
    s.absarc(x - r, -y + r, r, -math.pi / 2, 0.0, False)
    s.line_to(x, y - r)
    s.absarc(x - r, y - r, r, 0.0, math.pi / 2, False)
    s.line_to(-x + r, y)
    s.absarc(-x + r, y - r, r, math.pi / 2, math.pi, False)
    s.line_to(-x, -y + r)
    s.absarc(-x + r, -y + r, r, math.pi, 1.5 * math.pi, False)
    return s


class Beaker:
    def __init__(self, spec):
        self.spec = spec

    def geometry(self):
        b, f = self.spec, 0.001
        prof = [(0.0, 0.0, 0.0, -1.0), (b.radius - f, 0.0, 0.0, -1.0)]
        prof += [(b.radius - f + f * math.cos(a), f + f * math.sin(a), math.cos(a), math.sin(a))
                 for a in np.linspace(-math.pi / 2, 0.0, 6)]
        prof += [(b.radius, b.height, 1.0, 0.0)]
        prof += rounded_rim(b.r_in, b.radius, b.height)
        prof += [(b.r_in, b.floor, -1.0, 0.0), (b.r_in, b.floor, 0.0, 1.0), (0.0, b.floor, 0.0, 1.0)]
        return lathe(prof)


class Rack:
    def __init__(self, spec):
        self.spec = spec

    def geometries(self):
        r = self.spec
        s = rounded_rect_shape(r.width, r.length, r.corner_r)
        holes = []
        x0 = -r.width / 2 + (r.width - (r.rows - 1) * r.pitch) / 2
        for i in range(r.rows):
            for j in range(r.cols):
                h = tp.Path()
                h.absarc(x0 + i * r.pitch, (j - (r.cols - 1) / 2) * r.pitch, 0.5 * r.well_d, 0, 2 * math.pi, True)
                holes.append(h)
        s.holes = holes
        block = tp.ExtrudeGeometry(s, depth=r.height - r.floor, bevel_enabled=False, curve_segments=24)
        block.rotate_x(-math.pi / 2)
        block.translate(0.0, r.floor, 0.0)
        floor = tp.ExtrudeGeometry(rounded_rect_shape(r.width, r.length, r.corner_r), depth=r.floor,
                                   bevel_enabled=False, curve_segments=24)
        floor.rotate_x(-math.pi / 2)
        return block, floor

    def build(self, scene, center_world):
        mat = Materials.plastic(self.spec.color, 0.5)
        colliders = []
        for g in self.geometries():
            m = tp.Mesh(g, mat)
            m.position.set(*center_world)
            m.cast_shadow = m.receive_shadow = True
            scene.add(m)
            p = g.get_attribute("position")
            idx = g.get_index()
            tri = p[np.asarray(idx).reshape(-1, 3)] if idx is not None else p.reshape(-1, 3, 3)
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            top = (n[:, 1] > 0.9 * np.linalg.norm(n, axis=1)) & (np.abs(tri[:, :, 1] - self.spec.height).max(1) < 1e-5)
            cg = tp.BufferGeometry()
            cg.set_attribute("position", (tri[~top].reshape(-1, 3) + np.asarray(center_world)).astype(np.float32))
            colliders.append(tp.Mesh(cg, mat))
        return colliders


class Tube:
    def __init__(self, spec):
        self.spec = spec

    def tube_profile(self, n=12):
        t = self.spec
        prof = []
        for a in np.linspace(-math.pi / 2, 0.0, n):
            nr, ny = math.cos(a) * t.dome, math.sin(a) * t.r_out
            prof.append((t.r_out * math.cos(a), t.y_cap + t.dome * math.sin(a), nr / math.hypot(nr, ny),
                         ny / math.hypot(nr, ny)))
        prof += [(t.r_out, t.y_top, 1.0, 0.0)]
        prof += rounded_rim(t.r_in, t.r_out, t.y_top, 6)
        inner = t.dome - t.wall
        for a in np.linspace(0.0, -math.pi / 2, n):
            prof.append((t.r_in * math.cos(a), t.y_cap + inner * math.sin(a), -math.cos(a), -math.sin(a)))
        return prof

    def cap_profile(self):
        t = self.spec
        y0, y1, r, e = t.y_top - t.cap_skirt, t.cap_top - 0.5 * t.length, t.cap_r, 0.0012
        prof = [(t.r_out, y0, 0.0, -1.0), (r, y0, 0.0, -1.0), (r, y0, 1.0, 0.0), (r, y1 - e, 1.0, 0.0)]
        prof += [(r - e + e * math.cos(a), y1 - e + e * math.sin(a), math.cos(a), math.sin(a))
                 for a in np.linspace(0.0, math.pi / 2, 6)]
        prof += [(0.0, y1, 0.0, 1.0)]
        return prof

    def meshes(self):
        tube, cap = tp.Group(), tp.Group()
        wall = tp.Mesh(lathe(self.tube_profile()), Materials.glass(0.0009, ior=1.3, thin_walled=True))
        wall.cast_shadow = False
        tube.add(wall)
        c = tp.Mesh(lathe(self.cap_profile(), 64), Materials.plastic(self.spec.cap_color, 0.55))
        c.cast_shadow = c.receive_shadow = True
        cap.add(c)
        return tube, cap

    def collider_points(self):
        t = self.spec
        a = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
        ring = np.stack([np.cos(a), np.zeros_like(a), np.sin(a)], axis=1)
        tube = [ring * [t.r_out * math.cos(b), 0.0, t.r_out * math.cos(b)] + [0.0, t.y_cap + t.dome * math.sin(b), 0.0]
                for b in np.linspace(-math.pi / 2, 0.0, 8)]
        tube += [ring * [t.r_out, 0.0, t.r_out] + [0.0, y, 0.0] for y in np.linspace(t.y_cap, t.y_top, 6)]
        tube = np.unique(np.round(np.vstack(tube), 6), axis=0)
        cap_top = t.cap_top - 0.5 * t.length
        cap = np.vstack([ring * [t.cap_r, 0.0, t.cap_r] + [0.0, y, 0.0]
                         for y in np.linspace(t.y_top + 0.0002, cap_top, 4)])
        capped = np.vstack([tube, cap * [t.r_out / t.cap_r, 1.0, t.r_out / t.cap_r]])
        return tube, cap, capped


class RobotVisuals:
    def __init__(self, robot, hand_scale):
        self.robot = robot
        self.hand_scale = hand_scale

    def link(self, name):
        model = tp.ModelLoader().load(str(self.robot.mesh_path(name)))
        mat = Materials.standard(0x6f7275 if name in ("link_0", "link_7") else 0xff6c0a, 0.45, 0.1)
        model.traverse(lambda o: o.set_material(mat) if isinstance(o, tp.Mesh) else None)
        model.traverse(lambda o: setattr(o, "cast_shadow", True) if isinstance(o, tp.Mesh) else None)
        g = tp.Group()
        g.add(model)
        return g

    def hand_part(self, files):
        looks = {"nylon": (0x3c3d40, 0.75, 0.0), "steel": (0xb4b8bd, 0.3, 0.85), "red": (0xc8262b, 0.5, 0.0)}
        obj, scaled = tp.Group(), tp.Group()
        k = 0.001 * self.hand_scale
        scaled.scale.set(k, k, k)
        obj.add(scaled)
        for f in files:
            model = tp.ModelLoader().load(str(self.robot.hand_meshes / f"{f}.obj"))
            mat = Materials.standard(*looks[f.split("__")[1]])
            model.traverse(lambda o: o.set_material(mat) if isinstance(o, tp.Mesh) else None)
            model.traverse(lambda o: setattr(o, "cast_shadow", True) if isinstance(o, tp.Mesh) else None)
            scaled.add(model)
        return obj


class Adapter:
    def __init__(self, flange_z=0.045, plate_t=0.010, post_r=0.018, post_end=0.1645):
        self.flange_z = flange_z
        self.plate_t = plate_t
        self.post_r = post_r
        self.post_end = post_end
        self.bracket_x = (-0.02395, -0.01795)
        self.bracket_z = (0.1300, 0.1745)
        self.bracket_y = 0.0245
        self.clamp_z = (0.1405, post_end)
        self.clamp_r = 0.0235

    @staticmethod
    def _knurled_circle(r, ridges, depth):
        s = tp.Shape()
        a = np.linspace(0.0, 2.0 * math.pi, ridges * 6, endpoint=False)
        rr = r - depth * (0.5 - 0.5 * np.cos(ridges * a))
        s.move_to(rr[0] * math.cos(a[0]), rr[0] * math.sin(a[0]))
        for ri, ai in zip(rr[1:], a[1:]):
            s.line_to(ri * math.cos(ai), ri * math.sin(ai))
        s.close_path()
        return s

    @staticmethod
    def _polygon(pts, shape=False):
        p = tp.Shape() if shape else tp.Path()
        p.move_to(*pts[0])
        for q in pts[1:]:
            p.line_to(*q)
        p.close_path()
        return p

    def _hexagon(self, sw, shape=False):
        rc = sw / math.sqrt(3.0)
        return self._polygon([(rc * math.cos(k * math.pi / 3), rc * math.sin(k * math.pi / 3)) for k in range(6)], shape)

    @staticmethod
    def _extrude(shape, depth, bevel):
        g = tp.ExtrudeGeometry(shape, depth=depth - 2.0 * bevel, bevel_enabled=bevel > 0.0, curve_segments=24,
                               bevel_thickness=bevel, bevel_size=bevel, bevel_segments=1)
        g.translate(0.0, 0.0, bevel)
        return g

    def _screw(self, d, h, sw, steel, socket):
        c = 0.06 * d
        s = self._knurled_circle(0.5 * d - c, 36, 0.035 * d)
        s.holes = [self._hexagon(sw + 2.0 * c)]
        head = tp.Mesh(self._extrude(s, h, c), steel)
        bottom = tp.Mesh(tp.ShapeGeometry(self._hexagon(sw + 2.0 * c, shape=True)), socket)
        bottom.position.z = 0.35 * h
        g = tp.Group()
        for m in (head, bottom):
            m.cast_shadow = True
            g.add(m)
        return g

    def group(self):
        alu = Materials.standard(0xc4c8cd, 0.36, 0.9)
        black = Materials.standard(0x2a2c30, 0.42, 0.55)
        steel, socket = Materials.standard(0x2d2e31, 0.3, 0.85), Materials.standard(0x08080a, 0.8, 0.2)
        pin = Materials.standard(0xdadcdf, 0.18, 1.0)
        g = tp.Group()

        def add(mesh, T=None):
            if T is not None:
                set_pose(mesh, T)
            mesh.cast_shadow = mesh.receive_shadow = True
            g.add(mesh)
            return mesh

        fz, pt = self.flange_z, self.plate_t
        ro, c, cb = 0.0315, 0.0005, 0.002
        prof = [(0.0, 0.0, 0.0, -1.0), (ro - c, 0.0, 0.0, -1.0), (ro - c, 0.0, 0.7, -0.7), (ro, c, 0.7, -0.7),
                (ro, c, 1.0, 0.0), (ro, pt - c, 1.0, 0.0), (ro, pt - c, 0.7, 0.7), (ro - c, pt, 0.7, 0.7),
                (ro - c, pt, 0.0, 1.0), (ro - 0.001, pt, 0.0, 1.0), (ro - 0.001, pt, -1.0, 0.0),
                (ro - 0.001, pt - cb, -1.0, 0.0), (ro - 0.001, pt - cb, 0.0, 1.0), (0.0, pt - cb, 0.0, 1.0)]
        plate = lathe(prof, 128)
        plate.rotate_x(math.pi / 2)
        add(tp.Mesh(plate, alu)).position.z = fz
        bolt = [(0.025 * math.cos(math.radians(45 * k)), 0.025 * math.sin(math.radians(45 * k))) for k in range(8)]
        layer = tp.Shape()
        layer.absarc(0.0, 0.0, ro - 0.001, 0.0, 2 * math.pi, False)
        holes = []
        for x, y in bolt[1:]:
            h = tp.Path()
            h.absarc(x, y, 0.0055, 0.0, 2 * math.pi, True)
            holes.append(h)
        hp = tp.Path()
        hp.absarc(*bolt[0], 0.0031, 0.0, 2 * math.pi, True)
        layer.holes = holes + [hp]
        add(tp.Mesh(tp.ExtrudeGeometry(layer, depth=cb, bevel_enabled=False, curve_segments=48), alu)).position.z = \
            fz + pt - cb
        for x, y in bolt[1:]:
            add(self._screw(0.010, 0.006, 0.005, steel, socket), pose(np.eye(3), (x, y, fz + pt - cb)))
        dowel = tp.Mesh(tp.CylinderGeometry(0.003, 0.003, cb + 0.0003, 32), pin)
        dowel.rotation.x = math.pi / 2
        dowel.position.set(*bolt[0], fz + pt - 0.5 * cb + 0.00015)
        add(dowel)

        z0, z1, pr = fz + pt - cb, self.post_end, self.post_r
        post = lathe([(pr, 0.0, 1.0, 0.0), (pr, z1 - z0 - 0.0008, 1.0, 0.0), (pr, z1 - z0 - 0.0008, 0.7, 0.7),
                      (pr - 0.0008, z1 - z0, 0.7, 0.7), (pr - 0.0008, z1 - z0, 0.0, 1.0), (0.0, z1 - z0, 0.0, 1.0)], 96)
        post.rotate_x(math.pi / 2)
        add(tp.Mesh(post, alu)).position.z = z0

        gap, ear, ear_x, ri, cr = 0.0006, 0.0075, 0.0305, pr - 0.0004, self.clamp_r
        a0, b0 = math.atan2(ear, math.sqrt(cr ** 2 - ear ** 2)), math.asin(gap / ri)
        outer = [(cr * math.cos(a), cr * math.sin(a)) for a in np.linspace(a0, 2 * math.pi - a0, 96)]
        inner = [(ri * math.cos(a), ri * math.sin(a)) for a in np.linspace(2 * math.pi - b0, b0, 96)]
        outline = [(ear_x, gap), (ear_x, ear)] + outer + [(ear_x, -ear), (ear_x, -gap)] + inner
        clamp = add(tp.Mesh(self._extrude(self._polygon(outline, shape=True), self.clamp_z[1] - self.clamp_z[0],
                                          0.0004), black))
        clamp.position.z = self.clamp_z[0]
        bz, bx = self.bracket_z, self.bracket_x
        plate_s = rounded_rect_shape(2 * self.bracket_y - 0.0008, bz[1] - bz[0] - 0.0008, 0.003)
        add(tp.Mesh(self._extrude(plate_s, bx[1] - bx[0], 0.0004), black),
            pose(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (bx[0], 0.0, 0.5 * (bz[0] + bz[1]))))
        Rx = rot_axis([0.0, 1.0, 0.0], math.pi / 2)
        for y in (-0.0185, 0.0185):
            for z in (0.1345, 0.1700):
                add(self._screw(0.007, 0.004, 0.003, steel, socket), pose(Rx, (bx[1], y, z)))
        Ry = rot_axis([1.0, 0.0, 0.0], -math.pi / 2)
        add(self._screw(0.007, 0.004, 0.003, steel, socket),
            pose(Ry, (0.5 * (ear_x + cr), ear, 0.5 * sum(self.clamp_z))))
        g.rotation.z = math.pi
        return g
