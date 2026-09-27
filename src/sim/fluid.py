import math
from types import SimpleNamespace

import numpy as np
import threepp as tp
import warp as wp


class Fluid:
    def __init__(self, settings, device=None):
        wp.init()
        wp.set_module_options({"fast_math": True})
        self.s = settings
        self.device = device or wp.get_preferred_device()
        fs = settings.fluid
        self.n = settings.particles_per_tube()
        self.d = fs.spacing
        self.h = 2.0 * self.d
        self.cell = self.d
        self.iso = 0.5
        self.substeps = fs.substeps
        self.iterations = fs.iterations
        self.nb_radius = 1.1 * self.h
        self.max_nb = 64
        rho_cheb = 0.8
        self.omegas = [1.0]
        for k in range(1, self.iterations):
            self.omegas.append(2.0 / (2.0 - rho_cheb ** 2) if k == 1 else 4.0 / (4.0 - rho_cheb ** 2 * self.omegas[-1]))
        self.k = self._kernels()
        self._alloc()
        self._lattices()
        self.graph = None
        print(f"liquid: {settings.tube.fill_ml:.1f} ml per tube = {self.n:,} particles on {self.device}")


    def _pbf_constants(self, mass, s_corr_dq, s_corr_n):
        d, h = self.d, self.h
        poly6 = 315.0 / (64.0 * math.pi * h ** 9)
        spiky = -45.0 / (math.pi * h ** 6)
        reach = int(math.ceil(h / d)) + 1
        off = np.arange(-reach, reach + 1) * d
        gx, gy, gz = np.meshgrid(off, off, off, indexing="ij")
        r2 = (gx ** 2 + gy ** 2 + gz ** 2).ravel()
        r2 = r2[r2 < h * h]
        rho0 = float(mass * (poly6 * np.maximum(h * h - r2, 0.0) ** 3).sum())
        w_dq = float(poly6 * (h * h - s_corr_dq ** 2) ** 3)
        r = np.sqrt(r2[r2 > 1e-12])
        g = (mass / rho0) * abs(spiky) * (h - r) ** 2
        sum_grad2 = float((g ** 2).sum())
        ratio_typ = (poly6 * (h * h - d * d) ** 3) / w_dq
        return poly6, spiky, rho0, w_dq, 0.05 * sum_grad2, 0.30 * (0.10 / sum_grad2) / ratio_typ ** s_corr_n

    def _kernels(self):
        s, t, bk, fs = self.s, self.s.tube, self.s.beaker, self.s.fluid
        D, H, CELL = self.d, self.h, self.cell
        H6 = H ** 6
        MASS = 1.0
        POLY6, SPIKY, RHO0, W_DQ, EPS_CFM, S_CORR_K = self._pbf_constants(MASS, 0.20 * H, 4.0)
        R_OUT, R_IN, R_MID, Y_TOP, Y_CAP, CAP_K = t.r_out, t.r_in, t.r_mid, t.y_top, t.y_cap, t.cap_k
        RB_OUT, RB_IN, RB_MID, BEAKER_H = bk.radius, bk.r_in, bk.r_mid, bk.height
        WALL_EPS = 0.5 * D
        MAX_DP = 0.35 * D
        NB_RADIUS, MAX_NB = self.nb_radius, self.max_nb
        GRAVITY = -9.81
        JACOBI_RELAX = 0.4
        V_MAX, IMPACT_DAMP, FLOOR_FRICTION = fs.v_max, fs.impact_damp, fs.floor_friction
        THIN_N, THIN_BOOST = float(fs.thin_n), fs.thin_boost
        BAND = 0.002
        BUBBLE_RISE, BUBBLE_R = fs.bubble_rise, fs.bubble_r

        @wp.func
        def w_poly6(r2: float) -> float:
            d = H * H - r2
            if d <= 0.0:
                return 0.0
            return POLY6 * d * d * d

        @wp.func
        def w_spiky_grad(rv: wp.vec3, r: float) -> wp.vec3:
            if r <= 1.0e-9 or r >= H:
                return wp.vec3(0.0, 0.0, 0.0)
            return rv * (SPIKY * (H - r) * (H - r) / r)

        @wp.func
        def tube_local(q: wp.vec3) -> wp.vec3:
            k = wp.where(q[1] < Y_CAP, CAP_K, 1.0)
            qs = wp.vec3(q[0], Y_CAP + (q[1] - Y_CAP) * k, q[2])
            c = wp.vec3(0.0, wp.max(qs[1], Y_CAP), 0.0)
            d = qs - c
            l = wp.max(wp.length(d), 1.0e-9)
            target = wp.max(l, R_OUT + WALL_EPS)
            if l < R_MID:
                target = wp.min(l, R_IN - WALL_EPS)
            if q[1] > Y_TOP:
                target = l
            r = c + d * (target / l)
            return wp.vec3(r[0], Y_CAP + (r[1] - Y_CAP) / k, r[2])

        @wp.func
        def beaker(p: wp.vec3, b: wp.vec4) -> wp.vec3:
            out = p
            if p[1] <= BEAKER_H:
                dx = p[0] - b[0]
                dz = p[2] - b[1]
                r = wp.sqrt(dx * dx + dz * dz)
                if r < RB_MID:
                    s = wp.min(1.0, (RB_IN - WALL_EPS) / wp.max(r, 1.0e-9))
                    out = wp.vec3(b[0] + dx * s, wp.max(p[1], b[2] + WALL_EPS), b[1] + dz * s)
                elif r < RB_OUT + WALL_EPS:
                    s = (RB_OUT + WALL_EPS) / r
                    out = wp.vec3(b[0] + dx * s, p[1], b[1] + dz * s)
            return out

        @wp.func
        def collide(p: wp.vec3, piv: wp.vec3, rot: wp.mat33, b: wp.vec4) -> wp.vec3:
            q = tube_local(wp.transpose(rot) @ (p - piv))
            w = beaker(piv + rot @ q, b)
            return wp.vec3(w[0], wp.max(w[1], WALL_EPS), w[2])

        @wp.kernel(module="unique")
        def predict(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), xp: wp.array(dtype=wp.vec3), dt: float,
                    pivs: wp.array(dtype=wp.vec3), rots: wp.array(dtype=wp.mat33), bstate: wp.array(dtype=wp.vec4),
                    s: int):
            i = wp.tid()
            vi = v[i] + wp.vec3(0.0, GRAVITY, 0.0) * dt
            sp = wp.length(vi)
            if sp > V_MAX:
                vi = vi * (V_MAX / sp)
            v[i] = vi
            xp[i] = collide(x[i] + vi * dt, pivs[s], rots[s], bstate[0])

        @wp.kernel(module="unique")
        def find_neighbors(xp: wp.array(dtype=wp.vec3), grid: wp.uint64, nbr: wp.array2d(dtype=int),
                           nbr_n: wp.array(dtype=int)):
            i = wp.hash_grid_point_id(grid, wp.tid())
            if i < 0:
                return
            p = xp[i]
            n = int(0)
            for j in wp.hash_grid_query(grid, p, NB_RADIUS):
                rv = p - xp[j]
                if j != i and n < MAX_NB and wp.dot(rv, rv) < NB_RADIUS * NB_RADIUS:
                    nbr[i, n] = j
                    n += 1
            nbr_n[i] = n

        @wp.kernel(module="unique")
        def solve_lambda(xp: wp.array(dtype=wp.vec3), grid: wp.uint64, nbr: wp.array2d(dtype=int),
                         nbr_n: wp.array(dtype=int), lam: wp.array(dtype=float)):
            i = wp.hash_grid_point_id(grid, wp.tid())
            if i < 0:
                return
            p = xp[i]
            rho = MASS * w_poly6(0.0)
            grad_i = wp.vec3(0.0, 0.0, 0.0)
            sum_grad2 = float(0.0)
            for m in range(nbr_n[i]):
                rv = p - xp[nbr[i, m]]
                r2 = wp.dot(rv, rv)
                if r2 < H * H:
                    rho += MASS * w_poly6(r2)
                    g = w_spiky_grad(rv, wp.sqrt(r2)) * (MASS / RHO0)
                    grad_i += g
                    sum_grad2 += wp.dot(g, g)
            sum_grad2 += wp.dot(grad_i, grad_i)
            c = wp.max(rho / RHO0 - 1.0, 0.0)
            lam[i] = -c / (sum_grad2 + EPS_CFM)

        @wp.kernel(module="unique")
        def solve_delta(xp: wp.array(dtype=wp.vec3), lam: wp.array(dtype=float), grid: wp.uint64,
                        nbr: wp.array2d(dtype=int), nbr_n: wp.array(dtype=int), omega: float,
                        prev: wp.array(dtype=wp.vec3), pivs: wp.array(dtype=wp.vec3), rots: wp.array(dtype=wp.mat33),
                        bstate: wp.array(dtype=wp.vec4), s: int, out: wp.array(dtype=wp.vec3)):
            i = wp.hash_grid_point_id(grid, wp.tid())
            if i < 0:
                return
            p = xp[i]
            li = lam[i]
            dp = wp.vec3(0.0, 0.0, 0.0)
            for m in range(nbr_n[i]):
                j = nbr[i, m]
                rv = p - xp[j]
                r2 = wp.dot(rv, rv)
                if r2 < H * H and r2 > 1.0e-12:
                    qq = w_poly6(r2) / W_DQ
                    q2 = qq * qq
                    dp += w_spiky_grad(rv, wp.sqrt(r2)) * (li + lam[j] - S_CORR_K * q2 * q2)
            d = dp * (MASS / RHO0 * JACOBI_RELAX)
            dl = wp.length(d)
            if dl > MAX_DP:
                d = d * (MAX_DP / dl)
            if omega == 1.0 or li == 0.0:
                out[i] = collide(p + d, pivs[s], rots[s], bstate[0])
            else:
                e = prev[i] + (p + d - prev[i]) * omega - p
                el = wp.length(e)
                if el > MAX_DP:
                    e = e * (MAX_DP / el)
                out[i] = collide(p + e, pivs[s], rots[s], bstate[0])

        @wp.func
        def cohesion_kernel(r: float) -> float:
            if r <= 0.0 or r >= H:
                return 0.0
            a = (H - r) * (H - r) * (H - r) * r * r * r
            if 2.0 * r <= H:
                a = 2.0 * a - H6 / 64.0
            return a * (64.0 / H6)

        @wp.kernel(module="unique")
        def cohesion(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), grid: wp.uint64,
                     nbr: wp.array2d(dtype=int), nbr_n: wp.array(dtype=int), dt: float, prm: wp.array(dtype=float)):
            i = wp.hash_grid_point_id(grid, wp.tid())
            if i < 0:
                return
            p = x[i]
            a = wp.vec3(0.0, 0.0, 0.0)
            for m in range(nbr_n[i]):
                rv = p - x[nbr[i, m]]
                r = wp.length(rv)
                if r > 1.0e-9:
                    a -= rv * (cohesion_kernel(r) / r)
            v[i] = v[i] + a * (prm[0] * dt)

        @wp.kernel(module="unique")
        def finalize(x: wp.array(dtype=wp.vec3), xp: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                     bstate: wp.array(dtype=wp.vec4), dt: float, pivs: wp.array(dtype=wp.vec3),
                     rots: wp.array(dtype=wp.mat33), s: int, prm: wp.array(dtype=float)):
            i = wp.tid()
            vi = (xp[i] - x[i]) * (1.0 / dt)
            vi = vi - (vi - v[i]) * IMPACT_DAMP
            p = xp[i]
            b = bstate[0]
            in_beaker = (p[0] - b[0]) * (p[0] - b[0]) + (p[2] - b[1]) * (p[2] - b[1]) < RB_MID * RB_MID
            floor = wp.where(in_beaker, b[2], 0.0) + WALL_EPS
            if p[1] < floor + 0.25 * D:
                vi = wp.vec3(vi[0] * (1.0 - FLOOR_FRICTION), vi[1], vi[2] * (1.0 - FLOOR_FRICTION))
            q = wp.transpose(rots[s]) @ (p - pivs[s])
            rq = wp.sqrt(q[0] * q[0] + q[2] * q[2])
            if prm[3] > 0.0 and q[1] < Y_TOP and rq > R_IN - 1.5 * D and rq < R_MID:
                v_glass = (pivs[s] - pivs[wp.max(s - 1, 0)]) * (1.0 / dt)
                vi = v_glass + (vi - v_glass) * (1.0 - prm[3])
            sp = wp.length(vi)
            if sp > V_MAX:
                vi = vi * (V_MAX / sp)
            v[i] = vi
            x[i] = xp[i]

        @wp.kernel(module="unique")
        def viscosity(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), grid: wp.uint64,
                      nbr: wp.array2d(dtype=int), nbr_n: wp.array(dtype=int), v_out: wp.array(dtype=wp.vec3),
                      prm: wp.array(dtype=float), which: int):
            i = wp.hash_grid_point_id(grid, wp.tid())
            if i < 0:
                return
            p = x[i]
            vi = v[i]
            dv = wp.vec3(0.0, 0.0, 0.0)
            for m in range(nbr_n[i]):
                j = nbr[i, m]
                rv = p - x[j]
                r2 = wp.dot(rv, rv)
                if r2 < H * H:
                    dv += (v[j] - vi) * w_poly6(r2)
            v_out[i] = vi + dv * (prm[which] * MASS / RHO0)

        @wp.kernel(module="unique")
        def tube_momentum(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), piv: wp.vec3, rot: wp.mat33,
                          mark: wp.array(dtype=int), remark: int, out: wp.array(dtype=wp.vec4)):
            i = wp.tid()
            if remark == 1:
                q = wp.transpose(rot) @ (x[i] - piv)
                mark[i] = wp.where(q[1] < Y_TOP and q[0] * q[0] + q[2] * q[2] < R_MID * R_MID, 1, 0)
            if mark[i] == 1:
                vi = v[i]
                wp.atomic_add(out, 0, wp.vec4(vi[0], vi[1], vi[2], 1.0))

        @wp.kernel(module="unique")
        def bubbles_live(bp: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                         grid: wp.uint64, n: int, dt: float, seed: int):
            i = wp.tid()
            p = bp[i]
            count = int(0)
            vs = wp.vec3(0.0, 0.0, 0.0)
            for j in wp.hash_grid_query(grid, p, H):
                if wp.length(x[j] - p) < H:
                    count += 1
                    vs += v[j]
            if count < 20:
                rng = wp.rand_init(seed, i)
                j = wp.randi(rng, 0, n)
                bp[i] = x[j] + wp.vec3(wp.randf(rng) - 0.5, wp.randf(rng) - 0.5, wp.randf(rng) - 0.5) * (0.5 * D)
            else:
                bp[i] = p + (vs / float(count) + wp.vec3(0.0, BUBBLE_RISE, 0.0)) * dt

        @wp.kernel(module="unique")
        def bubbles_settled(bp: wp.array(dtype=wp.vec3), b: wp.vec4, top: float, dt: float, seed: int):
            i = wp.tid()
            p = bp[i]
            rng = wp.rand_init(seed, i)
            p = p + wp.vec3((wp.randf(rng) - 0.5) * 0.0004, BUBBLE_RISE * dt, (wp.randf(rng) - 0.5) * 0.0004)
            dx = p[0] - b[0]
            dz = p[2] - b[1]
            if p[1] > top or dx * dx + dz * dz > RB_IN * RB_IN:
                r = (RB_IN - 2.0 * BUBBLE_R) * wp.sqrt(wp.randf(rng))
                a = 6.2831853 * wp.randf(rng)
                p = wp.vec3(b[0] + r * wp.cos(a), b[2] + (top - b[2]) * wp.randf(rng), b[1] + r * wp.sin(a))
            bp[i] = p


        @wp.func
        def sample(field: wp.array3d(dtype=float), gx: float, gy: float, gz: float, nx: int, ny: int,
                   nz: int) -> float:
            i = int(wp.floor(gx))
            j = int(wp.floor(gy))
            k = int(wp.floor(gz))
            if i < 0 or j < 0 or k < 0 or i > nx - 2 or j > ny - 2 or k > nz - 2:
                return 0.0
            fx = gx - float(i)
            fy = gy - float(j)
            fz = gz - float(k)
            c00 = field[i, j, k] * (1.0 - fx) + field[i + 1, j, k] * fx
            c10 = field[i, j + 1, k] * (1.0 - fx) + field[i + 1, j + 1, k] * fx
            c01 = field[i, j, k + 1] * (1.0 - fx) + field[i + 1, j, k + 1] * fx
            c11 = field[i, j + 1, k + 1] * (1.0 - fx) + field[i + 1, j + 1, k + 1] * fx
            return (c00 * (1.0 - fy) + c10 * fy) * (1.0 - fz) + (c01 * (1.0 - fy) + c11 * fy) * fz

        @wp.kernel(module="unique")
        def splat(x: wp.array(dtype=wp.vec3), nbr_n: wp.array(dtype=int), field: wp.array3d(dtype=float),
                  origin: wp.vec3, nx: int, ny: int, nz: int, piv: wp.vec3, rot: wp.mat33, in_tube_grid: int):
            t = wp.tid()
            q = wp.transpose(rot) @ (x[t] - piv)
            r = wp.sqrt(q[0] * q[0] + q[2] * q[2])
            inside = r < R_MID and q[1] < Y_TOP + BAND
            band = r < R_MID and q[1] > Y_TOP - BAND and q[1] < Y_TOP + BAND
            p = x[t]
            if in_tube_grid == 1:
                if not inside:
                    return
                p = q
            elif inside and not band:
                return
            thin = wp.max(1.0 - float(nbr_n[t]) / THIN_N, 0.0)
            boost = 1.0 + THIN_BOOST * thin * thin
            gx = (p[0] - origin[0]) / CELL
            gy = (p[1] - origin[1]) / CELL
            gz = (p[2] - origin[2]) / CELL
            i = int(wp.floor(gx))
            j = int(wp.floor(gy))
            k = int(wp.floor(gz))
            if i < 0 or j < 0 or k < 0 or i > nx - 2 or j > ny - 2 or k > nz - 2:
                return
            fx = gx - float(i)
            fy = gy - float(j)
            fz = gz - float(k)
            for a in range(2):
                wa = wp.where(a == 0, 1.0 - fx, fx)
                for b in range(2):
                    wb = wp.where(b == 0, 1.0 - fy, fy)
                    for c in range(2):
                        wc = wp.where(c == 0, 1.0 - fz, fz)
                        wp.atomic_add(field, i + a, j + b, k + c, wa * wb * wc * boost)

        @wp.kernel(module="unique")
        def blur_axis(src: wp.array3d(dtype=float), dst: wp.array3d(dtype=float), axis: int, w: float, nx: int,
                      ny: int, nz: int):
            i, j, k = wp.tid()
            di = wp.where(axis == 0, 1, 0)
            dj = wp.where(axis == 1, 1, 0)
            dk = wp.where(axis == 2, 1, 0)
            lo = src[wp.max(i - di, 0), wp.max(j - dj, 0), wp.max(k - dk, 0)]
            hi = src[wp.min(i + di, nx - 1), wp.min(j + dj, ny - 1), wp.min(k + dk, nz - 1)]
            dst[i, j, k] = w * lo + (1.0 - 2.0 * w) * src[i, j, k] + w * hi

        @wp.kernel(module="unique")
        def expand(verts: wp.array(dtype=wp.vec3), indices: wp.array(dtype=wp.int32), field: wp.array3d(dtype=float),
                   origin: wp.vec3, nx: int, ny: int, nz: int, piv: wp.vec3, rot: wp.mat33,
                   out_pos: wp.array(dtype=wp.vec3), out_nrm: wp.array(dtype=wp.vec3)):
            t = wp.tid()
            for c in range(3):
                p = verts[indices[t * 3 + c]]
                gx = (p[0] - origin[0]) / CELL
                gy = (p[1] - origin[1]) / CELL
                gz = (p[2] - origin[2]) / CELL
                g = wp.vec3(sample(field, gx + 1.0, gy, gz, nx, ny, nz) - sample(field, gx - 1.0, gy, gz, nx, ny, nz),
                            sample(field, gx, gy + 1.0, gz, nx, ny, nz) - sample(field, gx, gy - 1.0, gz, nx, ny, nz),
                            sample(field, gx, gy, gz + 1.0, nx, ny, nz) - sample(field, gx, gy, gz - 1.0, nx, ny, nz))
                l = wp.length(g)
                o = t * 3 + (3 - c) % 3
                out_pos[o] = piv + rot @ p
                out_nrm[o] = rot @ wp.where(l > 1.0e-9, -g / l, wp.vec3(0.0, 1.0, 0.0))

        return SimpleNamespace(**{k: v for k, v in locals().items() if isinstance(v, wp.Kernel)})


    def _alloc(self):
        n, dev = self.n, self.device
        self.x = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.v = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.xp = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.xa = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.xb = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.vtmp = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.lam = wp.zeros(n, dtype=float, device=dev)
        self.nbr = wp.zeros((n, self.max_nb), dtype=int, device=dev)
        self.nbr_n = wp.zeros(n, dtype=int, device=dev)
        self.grid = wp.HashGrid(64, 64, 64, dev)
        self.grid.reserve(n)
        self.rots = wp.zeros(self.substeps, dtype=wp.mat33, device=dev)
        self.pivs = wp.zeros(self.substeps, dtype=wp.vec3, device=dev)
        self.bstate = wp.zeros(1, dtype=wp.vec4, device=dev)
        self.prm = wp.zeros(4, dtype=float, device=dev)
        self.mark = wp.zeros(n, dtype=int, device=dev)
        self.msum = wp.zeros(1, dtype=wp.vec4, device=dev)
        self.bubbles = wp.zeros(max(1, max(q.bubbles for q in self.s.layout.liquids)), dtype=wp.vec3, device=dev)

    def _lattices(self):
        t, bk, d = self.s.tube, self.s.beaker, self.d
        wall_eps = 0.5 * d
        a = np.arange(-t.r_in, t.r_in + d, d)
        lx, ly, lz = np.meshgrid(a, np.arange(t.y_cap - t.r_in, t.y_top, d), a, indexing="ij")
        lat = np.stack([lx.ravel(), ly.ravel(), lz.ravel()], axis=-1)
        lim = t.r_in - wall_eps
        stretched = lat * [1.0, t.cap_k, 1.0] - [0.0, t.y_cap * t.cap_k, 0.0]
        self.tube_lattice = lat[np.where(lat[:, 1] >= t.y_cap, np.hypot(lat[:, 0], lat[:, 2]) <= lim,
                                         np.linalg.norm(stretched, axis=1) <= lim)]
        b = np.arange(-bk.r_in, bk.r_in + d, d)
        bx, by, bz = np.meshgrid(b, np.arange(bk.floor + wall_eps, bk.height, d), b, indexing="ij")
        lat = np.stack([bx.ravel(), by.ravel(), bz.ravel()], axis=-1)
        lat = lat[np.hypot(lat[:, 0], lat[:, 2]) <= bk.r_in - wall_eps]
        self.beaker_lattice = lat[np.argsort(lat[:, 1], kind="stable")]

    def fill_tube(self, T):
        pts = self.tube_lattice @ T[:3, :3].T + T[:3, 3]
        return pts[np.argsort(pts[:, 1], kind="stable")[:self.n]].astype(np.float32)

    def fill_beaker(self, center, count):
        return (self.beaker_lattice[:count] + center).astype(np.float32)

    def start(self, points, liquid, beaker_xz, floor):
        self.x.assign(points)
        self.v.zero_()
        self.nbr_n.zero_()
        self.bstate.assign(np.array([[beaker_xz[0], beaker_xz[1], floor, 0.0]], np.float32))
        self.prm.assign(np.array([liquid.cohesion, liquid.visc, liquid.visc_sub, liquid.wall], np.float32))

    def _launch_frame(self):
        k, n, dev, dt = self.k, self.n, self.device, self.s.fluid.dt_sub
        x, v = self.x, self.v
        for s in range(self.substeps):
            wp.launch(k.cohesion, dim=n, inputs=[x, v, self.grid.id, self.nbr, self.nbr_n, dt, self.prm], device=dev)
            wp.launch(k.predict, dim=n, inputs=[x, v, self.xp, dt, self.pivs, self.rots, self.bstate, s], device=dev)
            self.grid.build(points=self.xp, radius=self.nb_radius)
            wp.launch(k.find_neighbors, dim=n, inputs=[self.xp, self.grid.id, self.nbr, self.nbr_n], device=dev)
            prev = cur = self.xp
            spare = [self.xa, self.xb]
            for it in range(self.iterations):
                out = spare.pop(0)
                wp.launch(k.solve_lambda, dim=n, inputs=[cur, self.grid.id, self.nbr, self.nbr_n, self.lam], device=dev)
                wp.launch(k.solve_delta, dim=n, device=dev,
                          inputs=[cur, self.lam, self.grid.id, self.nbr, self.nbr_n, self.omegas[it], prev, self.pivs,
                                  self.rots, self.bstate, s, out])
                if prev is not cur:
                    spare.append(prev)
                prev, cur = cur, out
            wp.launch(k.finalize, dim=n, device=dev,
                      inputs=[x, cur, v, self.bstate, dt, self.pivs, self.rots, s, self.prm])
            wp.launch(k.viscosity, dim=n, inputs=[x, v, self.grid.id, self.nbr, self.nbr_n, self.vtmp, self.prm, 2],
                      device=dev)
            wp.copy(v, self.vtmp)
        wp.launch(k.viscosity, dim=n, inputs=[x, v, self.grid.id, self.nbr, self.nbr_n, self.vtmp, self.prm, 1],
                  device=dev)
        wp.copy(v, self.vtmp)

    def step(self, tube_poses):
        self.rots.assign(tube_poses[:, :3, :3].astype(np.float32))
        self.pivs.assign(tube_poses[:, :3, 3].astype(np.float32))
        if self.graph is None:
            wp.load_module(device=self.device)
            with wp.ScopedCapture(self.device) as cap:
                self._launch_frame()
            self.graph = cap.graph
        wp.capture_launch(self.graph)

    def momentum_in_tube(self, T, remark):
        self.msum.zero_()
        wp.launch(self.k.tube_momentum, dim=self.n, device=self.device,
                  inputs=[self.x, self.v, *wp_pose(T), self.mark, remark, self.msum])
        m = self.s.physics.liquid_density * self.d ** 3
        s = self.msum.numpy()[0]
        return m * s[:3], m * s[3]

    def step_bubbles(self, n, seed, settled_beaker=None, top=0.0):
        if settled_beaker is None:
            wp.launch(self.k.bubbles_live, dim=n, device=self.device,
                      inputs=[self.bubbles, self.x, self.v, self.grid.id, self.n, self.s.fluid.dt, seed])
        else:
            wp.launch(self.k.bubbles_settled, dim=n, device=self.device,
                      inputs=[self.bubbles, wp.vec4(*settled_beaker, 0.0), top, self.s.fluid.dt, seed])
        return self.bubbles.numpy()[:n]

    def split_ml(self, T_tube, beaker_center, beaker_count):
        t, bk = self.s.tube, self.s.beaker
        p = self.x.numpy()
        q = (p - T_tube[:3, 3]) @ T_tube[:3, :3]
        in_tube = (q[:, 1] < t.y_top) & (np.hypot(q[:, 0], q[:, 2]) < t.r_mid)
        in_b = (p[:, 1] < bk.height) & (np.hypot(p[:, 0] - beaker_center[0], p[:, 2] - beaker_center[2]) < bk.r_mid)
        ml = self.d ** 3 * 1e6
        return in_tube.sum() * ml, (in_b.sum() + beaker_count) * ml, int(in_b.sum())


def wp_pose(T):
    return wp.vec3(*T[:3, 3]), wp.mat33(*T[:3, :3].ravel())


class Surface:
    def __init__(self, fluid, dims, max_tris, in_tube_grid):
        self.f = fluid
        self.dims = dims
        self.max_tris = max_tris
        self.in_tube_grid = in_tube_grid
        dev = fluid.device
        self.field = wp.zeros(dims, dtype=float, device=dev)
        self.tmp = wp.zeros(dims, dtype=float, device=dev)
        self.mc = wp.MarchingCubes(*dims, device=dev)
        self.geometry = tp.BufferGeometry()
        self.geometry.set_attribute("position", np.zeros((max_tris * 3, 3), np.float32))
        self.geometry.set_attribute("normal", np.tile(np.float32([0, 1, 0]), (max_tris * 3, 1)))
        self.geometry.set_draw_range(0, 3)
        self.gl = None
        self.vk = None
        self.mesh = None
        self.ntris = 0
        self._pending = None

    def register(self, renderer, mesh):
        self.mesh = mesh
        if isinstance(renderer, tp.GLRenderer):
            flags = wp.RegisteredGLBuffer.WRITE_DISCARD
            self.gl = [wp.RegisteredGLBuffer(int(renderer.gl_buffer_id(self.geometry, a)), self.f.device, flags)
                       for a in ("position", "normal")]
            return
        from threepp.cuda_interop import VkInteropArray
        h = renderer.enable_vertex_interop(mesh, self._vk_write, validate=True, stable_correspondence=False)
        if h is None:
            raise RuntimeError("the Vulkan renderer did not export the liquid's vertex buffers")
        n = self.max_tris * 3
        self.vk = [VkInteropArray(handle, size, wp.vec3, n, self.f.device) for handle, size in h]

    def release(self, renderer):
        if self.vk is not None:
            for a in self.vk:
                a.close()
            renderer.disable_vertex_interop(self.mesh)
            self.vk = None

    def _vk_write(self):
        if self._pending is not None:
            f, (nx, ny, nz) = self.f, self.dims
            a, o, ntris, piv, rot = self._pending
            wp.launch(f.k.expand, dim=ntris, device=f.device,
                      inputs=[self.mc.verts, self.mc.indices, a, o, nx, ny, nz, *self._out_frame(piv, rot),
                              self.vk[0].array, self.vk[1].array])
            self._pending = None
        wp.synchronize_device(self.f.device)

    def hide(self):
        self.ntris = 0
        self._pending = None
        self.geometry.set_draw_range(0, 0)
        if self.vk is not None:
            self.mesh.visible = False

    def _build(self, pts, counts, origin, piv, rot):
        f, (nx, ny, nz) = self.f, self.dims
        o = wp.vec3(*origin)
        self.mc.domain_bounds_lower_corner = o
        self.mc.domain_bounds_upper_corner = wp.vec3(*[origin[i] + (self.dims[i] - 1) * f.cell for i in range(3)])
        self.field.zero_()
        wp.launch(f.k.splat, dim=pts.shape[0], device=f.device,
                  inputs=[pts, counts, self.field, o, nx, ny, nz, piv, rot, self.in_tube_grid])
        a, b = self.field, self.tmp
        for w in (0.25, 0.125):
            for axis in range(3):
                wp.launch(f.k.blur_axis, dim=self.dims, inputs=[a, b, axis, w, nx, ny, nz], device=f.device)
                a, b = b, a
        self.mc.surface(a, f.iso)
        return a, o, min(self.mc.indices.shape[0] // 3, self.max_tris)

    def _out_frame(self, piv, rot):
        if self.in_tube_grid:
            return piv, rot
        return wp.vec3(0.0, 0.0, 0.0), wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def update(self, origin, T):
        f, (nx, ny, nz) = self.f, self.dims
        piv, rot = wp_pose(T)
        a, o, ntris = self._build(f.x, f.nbr_n, origin, piv, rot)
        if self.vk is not None:
            self._pending = (a, o, ntris, piv, rot) if ntris > 0 else None
        elif ntris > 0:
            dp = self.gl[0].map(dtype=wp.vec3, shape=(self.max_tris * 3,))
            dn = self.gl[1].map(dtype=wp.vec3, shape=(self.max_tris * 3,))
            wp.launch(f.k.expand, dim=ntris, device=f.device,
                      inputs=[self.mc.verts, self.mc.indices, a, o, nx, ny, nz, *self._out_frame(piv, rot), dp, dn])
            self.gl[0].unmap()
            self.gl[1].unmap()
        self.ntris = ntris
        self.geometry.set_draw_range(0, 3 * ntris)
        if self.vk is not None:
            self.mesh.visible = ntris > 0

    def mesh_arrays(self, origin, T, points=None):
        f, (nx, ny, nz) = self.f, self.dims
        piv, rot = wp_pose(T)
        if points is None:
            pts, counts = f.x, f.nbr_n
        else:
            pts = wp.array(points, dtype=wp.vec3, device=f.device)
            counts = wp.full(len(points), 64, dtype=int, device=f.device)
        a, o, ntris = self._build(pts, counts, origin, piv, rot)
        if ntris == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
        dp = wp.zeros(ntris * 3, dtype=wp.vec3, device=f.device)
        dn = wp.zeros(ntris * 3, dtype=wp.vec3, device=f.device)
        wp.launch(f.k.expand, dim=ntris, device=f.device,
                  inputs=[self.mc.verts, self.mc.indices, a, o, nx, ny, nz, *self._out_frame(piv, rot), dp, dn])
        return dp.numpy(), dn.numpy()

    def bake(self, points, origin, T):
        pos, nrm = self.mesh_arrays(origin, T, points)
        g = tp.BufferGeometry()
        if len(pos):
            g.set_attribute("position", pos)
            g.set_attribute("normal", nrm)
        return g
