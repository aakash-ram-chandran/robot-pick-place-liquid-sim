from pathlib import Path

import numpy as np
import threepp as tp

from demo import Demo
from settings import Settings

ROOT = Path(__file__).resolve().parents[2]


class LiveApp:
    def __init__(self, settings):
        self.demo = Demo(settings, ROOT / "assets", ROOT / "cache" / "plan.npz",
                         title="robot liquid handling: decap, pour, recap")
        d = self.demo
        self.controls = tp.OrbitControls(d.camera, d.canvas)
        self.controls.enable_damping = True
        self.controls.target.set(*d.look)
        d.canvas.on_window_resize(self.on_resize)
        self.ui = tp.ImguiContext(d.canvas, d.renderer) if tp.HAS_IMGUI else None
        self.split = (0.0, 0.0)

    def on_resize(self, w, h):
        self.demo.camera.aspect = w / max(h, 1)
        self.demo.camera.update_projection_matrix()
        self.demo.renderer.set_size(w, h)

    def hud(self):
        d, im = self.demo, tp.imgui
        im.set_next_window_pos(10, 10)
        im.set_next_window_size(300, 0)
        im.begin("robot liquid handling")
        im.text(f"t {d.frame * d.s.fluid.dt:5.1f} / {d.n_frames * d.s.fluid.dt:.0f} s")
        im.text(d.label())
        k = d.active
        if k >= 0:
            b = d.s.layout.tube_beaker[k]
            im.text(f"in tube {self.split[0]:4.2f} ml   {d.s.layout.liquids[b].name} in beaker: {self.split[1]:4.2f} ml")
            im.text(f"cap {'off, in the fingertips' if d.physics.cap_off(k) else 'on'}"
                    f"   liquid on tube {np.linalg.norm(d.push) * 1000:4.0f} mN")
        im.text(f"{im.get_framerate():5.1f} fps")
        im.text("R = restart")
        im.end()

    def frame(self):
        d = self.demo
        if d.frame >= d.n_frames or d.canvas.is_key_down("R"):
            d.reset()
        d.step()
        d.place()
        if d.frame % 15 == 0:
            self.split = d.liquid_split()
        if self.ui is not None:
            self.controls.enabled = not self.ui.want_capture_mouse
        self.controls.update()
        d.render()
        if self.ui is not None:
            self.ui.render(self.hud)

    def run(self):
        self.demo.canvas.animate(self.frame)


if __name__ == "__main__":
    LiveApp(Settings()).run()
