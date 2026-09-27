import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from demo import Demo
from geometry import to_world
from settings import Settings

ROOT = Path(__file__).resolve().parents[2]


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


class VideoDirector:
    def __init__(self, settings, out: Path, width=1920, height=1080, supersample=1.334, fps=30, passes=8, blend=0.8,
                 orbit_s=2.5):
        self.size = (width, height)
        self.d = Demo(settings, ROOT / "assets", ROOT / "cache" / "plan.npz",
                      width=int(round(width * supersample)), height=int(round(height * supersample)),
                      renderer="vulkan", title="rendering")
        self.out = Path(out)
        self.frames = self.out.parent / f"{self.out.stem}_frames"
        self.fps = fps
        self.passes = passes
        self.blend = blend
        self.orbit_s = orbit_s
        self.center = np.array([0.50, 0.07, 0.0])
        self._setup_renderer()
        d, L = self.d, settings.layout
        self.pick = d.tubes_upright[0][:3, 3]
        self.place = to_world(d.planner.tube_in_rack(L.rack_b, L.place_slots[0]))[:3, 3]
        self.segments = self._segments()
        self.cam = None
        self.follow = None

    def _setup_renderer(self):
        r = self.d.renderer
        r.restir_di = True
        r.probe_gi = True
        r.deferred_ao = True
        r.denoise = True
        r.gbuffer_msaa = 4
        r.depth_of_field = False
        r.sun_angular_radius = 1.5
        r.tone_mapping_exposure = 1.25

    def _t(self, label):
        return self.d.plan.first(label) * self.d.s.fluid.dt_sub

    def _segments(self):
        L, t = self.d.s.layout, self._t
        end = (self.d.n_frames - 1) * self.d.s.fluid.dt
        b0 = L.tube_beaker[0]
        segs = [(0.0, t("tube 1: down over the tube"), 3.0, "wide"),
                (t("tube 1: down over the tube"), t("tube 1: lift") + 0.8, 1.0, "pick"),
                (t("tube 1: lift") + 0.8, t("tube 1: carry to beaker 1") - 0.3, 1.6, ("cap_front", 0)),
                (t("tube 1: carry to beaker 1") - 0.3, t("tube 1: pour") + 2.0, 4.0, ("to", b0)),
                (t("tube 1: pour") + 2.0, t("tube 1: pour") + 12.0, 1.6, ("pour_front", 0)),
                (t("tube 1: pour") + 12.0, t("tube 1: straighten up") + 0.6, 3.5, ("beaker", b0)),
                (t("tube 1: straighten up") + 0.6, t("tube 1: cap back over the tube") - 0.3, 3.5, "hand"),
                (t("tube 1: cap back over the tube") - 0.3, t("tube 1: fingertips home") + 0.3, 1.8, ("cap_front", 0)),
                (t("tube 1: fingertips home") + 0.3, t("tube 1: insert"), 4.0, "hand"),
                (t("tube 1: insert"), t("tube 1: release") + 2.2, 1.0, "place")]
        prev = t("tube 1: release") + 2.2
        for k in range(1, self.d.n):
            b = L.tube_beaker[k]
            pour = t(f"tube {k + 1}: pour")
            segs += [(prev, pour + 2.0, 16.0, "wide"),
                     (pour + 2.0, pour + 12.0, 2.0, ("pour_front", k)),
                     (pour + 12.0, t(f"tube {k + 1}: straighten up") + 0.6, 4.0, ("beaker", b))]
            prev = t(f"tube {k + 1}: straighten up") + 0.6
        segs.append((prev, end, 16.0, "wide"))
        return segs

    def length(self):
        return sum((b - a) / sp for a, b, sp, _ in self.segments) + self.orbit_s

    def _tube_top(self, k):
        T = self.d.physics.tube_pose(k)
        return T[:3, 3] + 0.05 * T[:3, 1]

    def _hand(self):
        ph = self.d.physics
        return ph.link_pose("link_7") @ ph.T7h

    def _shot(self, kind, tv):
        d = self.d
        if kind == "wide":
            a = math.radians(58.0) + 0.06 * tv
            return self.center + [0.86 * math.cos(a), 0.46, 0.86 * math.sin(a)], self.center.copy()
        if kind == "hand":
            tgt = d.physics.link_pose("bridge")[:3, 3]
            a = math.radians(35.0) + 0.10 * tv
            return tgt + [0.30 * math.cos(a), 0.10, 0.30 * math.sin(a)], tgt
        if kind == "pick":
            H = self._hand()
            tgt = d.physics.tube_pose(0)[:3, 3] + [0.0, 0.02, 0.0]
            r = 0.18 - 0.006 * tv
            return tgt + r * H[:3, 0] + 0.05 * H[:3, 1] + [0.0, 0.04, 0.0], tgt
        if kind == "place":
            tgt = self.place + [0.0, 0.02, 0.0]
            r = 0.22 - 0.008 * tv
            return tgt + [0.02, 0.035, r], tgt
        what, k = kind
        if what == "cap_front":
            H = self._hand()
            tgt = self._tube_top(k) + [0.0, 0.004, 0.0]
            side = 0.035 * math.sin(0.25 * tv + 0.4)
            return tgt + 0.15 * H[:3, 0] + side * H[:3, 1] + [0.0, 0.035, 0.0], tgt
        if what == "pour_front":
            tgt = self._tube_top(k)
            return tgt + [-0.17 + 0.002 * tv, 0.035, 0.09], tgt + [-0.01, -0.02, 0.0]
        c = d.beaker_center(k) + [0.0, 0.045, 0.0]
        r = 0.36 - 0.012 * tv if what == "beaker" else 0.40
        return c + [-0.04, 0.06, r], c + [0.02, 0.01, 0.0]

    def _camera(self, kind, tv, start):
        p, t = self._shot(kind, tv)
        if kind in ("hand", "pick") or kind[0] in ("cap_front", "pour_front"):
            self.follow = t if self.follow is None else self.follow + 0.15 * (t - self.follow)
            p, t = p - t + self.follow, self.follow
        if start is not None and tv < self.blend:
            w = ease(tv / self.blend)
            p, t = start[0] + w * (p - start[0]), start[1] + w * (t - start[1])
        return np.asarray(p), np.asarray(t)

    def _look(self, p, t):
        c = self.d.camera
        c.position.set(*p)
        c.look_at(*t)
        self.d.renderer.focus_distance = float(np.linalg.norm(t - p))
        self.cam = (p, t)

    def _save(self, path):
        for _ in range(self.passes - 1):
            self.d.render()
        self.d.save_frame(path)

    def stills(self, shots, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        d = self.d
        for when, kind in shots:
            while d.frame * d.s.fluid.dt < when:
                d.step()
            d.place()
            self.follow = None
            self._look(*self._camera(kind, 1.0, None))
            name = kind if isinstance(kind, str) else kind[0]
            self._save(out / f"{when:06.2f}s_{name}.png")
            print(f"  {name} at {when:.1f} s", flush=True)

    def run(self):
        d, sim_fps = self.d, self.d.s.fluid.fps
        if self.frames.exists():
            shutil.rmtree(self.frames)
        self.frames.mkdir(parents=True)
        print(f"{len(self.segments)} shots, {self.length():.1f} s of video")
        n = 0
        for i, (t0, t1, speed, kind) in enumerate(self.segments):
            start = self.cam
            self.follow = None
            stride = max(1, int(round(speed * sim_fps / self.fps)))
            k = 0
            while d.frame * d.s.fluid.dt < t1 and d.frame < d.n_frames:
                d.step()
                d.place()
                if k % stride == 0:
                    self._look(*self._camera(kind, k / stride / self.fps, start))
                    self._save(self.frames / f"{n:05d}.png")
                    n += 1
                k += 1
            print(f"  shot {i + 1}/{len(self.segments)} done, {n / self.fps:.1f} s", flush=True)
        start = self.cam
        for j in range(int(self.orbit_s * self.fps)):
            tv = j / self.fps
            a = math.radians(20.0) + 0.35 * tv
            p = self.center + [0.95 * math.cos(a), 0.45, 0.95 * math.sin(a)]
            w = ease(tv / self.blend)
            self._look(start[0] + w * (p - start[0]), start[1] + w * (self.center - start[1]))
            self._save(self.frames / f"{n:05d}.png")
            n += 1
        self.encode()

    def encode(self):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(self.fps), "-i",
                        str(self.frames / "%05d.png"), "-vf", f"scale={self.size[0]}:{self.size[1]}:flags=lanczos",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.out)], check=True)
        print(f"wrote {self.out}")

    def gif(self, path: Path, start=0.0, seconds=14.0, fps=12, width=640):
        f = f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160[p];[b][p]paletteuse=dither=sierra2_4a"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(seconds), "-i", str(self.out),
                        "-vf", f, "-loop", "0", str(path)], check=True)
        print(f"wrote {path}")


if __name__ == "__main__":
    v = VideoDirector(Settings(), ROOT / "media" / "demo.mp4")
    v.run()
    v.gif(ROOT / "media" / "demo.gif", start=1.2)
