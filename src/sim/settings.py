from dataclasses import dataclass, field


@dataclass
class Liquid:
    name: str
    color: int
    visc: float = 0.2
    visc_sub: float = 0.0
    wall: float = 0.0
    cohesion: float = 4.0
    tilt_slow: float = 2.0
    drain: float = 1.5
    bubbles: int = 0


def default_liquids():
    return [Liquid("water", 0x1f6fd0),
            Liquid("honey", 0xd98f1c, visc_sub=0.7, wall=0.2, cohesion=30.0, tilt_slow=1.0, drain=14.0),
            Liquid("soda", 0x8fc95a, bubbles=250)]


@dataclass
class Tube:
    od: float = 0.016
    length: float = 0.100
    wall: float = 0.0009
    dome: float = 0.00623
    cap_top: float = 0.1123
    cap_r: float = 0.00935
    cap_skirt: float = 0.0075
    cap_color: int = 0xe4ded2
    fill_ml: float = 3.0
    grasp_y: float = 0.076

    @property
    def r_out(self):
        return 0.5 * self.od

    @property
    def r_in(self):
        return self.r_out - self.wall

    @property
    def r_mid(self):
        return 0.5 * (self.r_in + self.r_out)

    @property
    def y_top(self):
        return 0.5 * self.length

    @property
    def y_cap(self):
        return -0.5 * self.length + self.dome

    @property
    def cap_k(self):
        return self.r_out / self.dome


@dataclass
class Rack:
    rows: int = 5
    cols: int = 10
    pitch: float = 0.022
    well_d: float = 0.020
    width: float = 0.120
    length: float = 0.230
    height: float = 0.045
    floor: float = 0.0054
    corner_r: float = 0.006
    color: int = 0x8e949c

    def slot_xy(self, center, slot):
        row, col = slot
        return (center[0] - self.width / 2 + (self.width - (self.rows - 1) * self.pitch) / 2 + row * self.pitch,
                center[1] - (col - (self.cols - 1) / 2) * self.pitch)


@dataclass
class Beaker:
    radius: float = 0.021
    height: float = 0.055
    wall: float = 0.0012
    floor: float = 0.0015

    @property
    def r_in(self):
        return self.radius - self.wall

    @property
    def r_mid(self):
        return self.radius - 0.5 * self.wall


@dataclass
class Layout:
    rack_a: tuple = (0.48, 0.24)
    rack_b: tuple = (0.48, -0.24)
    pick_slots: list = field(default_factory=lambda: [(4, 6), (4, 3), (2, 9), (2, 6), (2, 3), (2, 0), (0, 9),
                                                      (0, 6), (0, 3), (0, 0)])
    place_slots: list = field(default_factory=lambda: [(0, 3), (0, 6), (0, 9), (2, 3), (2, 6), (2, 9), (4, 0),
                                                       (4, 3), (4, 6), (4, 9)])
    beakers: list = field(default_factory=lambda: [(0.62, 0.05), (0.62, 0.0), (0.62, -0.05)])
    tube_beaker: list = field(default_factory=lambda: [0, 2])
    liquids: list = field(default_factory=default_liquids)

    @property
    def n_tubes(self):
        return len(self.tube_beaker)


@dataclass
class Hand:
    scale: float = 0.6
    bridge_cap: float = 0.0064
    jaw_open: float = 0.020
    jaw_tube: float = 0.0080
    jaw_cap: float = 0.0092
    jaw_ease: float = 0.0110
    finger_mid: float = 0.024
    twist_stroke: float = 0.012
    twist_strokes: int = 2
    cap_clear: float = 0.016
    grip_force: float = 15.0
    squeeze: float = 0.003
    press: float = 0.002
    release_lag: float = 0.12
    pad_friction: float = 1.0
    slide_stiffness: float = 5000.0
    slide_damping: float = 60.0


@dataclass
class Motion:
    q_home: list = field(default_factory=lambda: [0.0, 0.454, 0.0, -1.614, 0.0, 1.073, 0.0])
    approach: float = 0.06
    lift: float = 0.135
    vmax_joint: float = 0.8
    amax_joint: float = 2.0
    vmax_tool: float = 0.25
    amax_tool: float = 1.0
    vmax_carry: float = 0.15
    amax_carry: float = 0.5
    tilt_fast: float = 30.0
    pour_from: float = 82.0
    pour_to: float = 100.0
    tilt_max: float = 110.0
    front_d: float = 0.032
    lip_start: float = 0.054
    lip_in: float = 0.020
    lip_drop: float = 0.020
    ik_every: int = 60


@dataclass
class Physics:
    hz: int = 960
    arm_stiffness: float = 2.0e5
    arm_damping: float = 2.0e3
    arm_torque: list = field(default_factory=lambda: [320.0, 320.0, 176.0, 176.0, 110.0, 40.0, 40.0])
    arm_limits_deg: list = field(default_factory=lambda: [170.0, 120.0, 170.0, 120.0, 170.0, 120.0, 175.0])
    cap_friction: float = 0.03
    seal_pull: float = 8.0
    seat_pull: float = 40.0
    liquid_density: float = 1000.0


@dataclass
class Fluid:
    spacing: float = 0.0006
    substeps: int = 20
    iterations: int = 4
    fps: int = 60
    v_max: float = 1.2
    impact_damp: float = 0.5
    floor_friction: float = 0.05
    thin_boost: float = 14.0
    thin_n: int = 24
    max_tris: int = 300_000
    max_tris_tube: int = 60_000
    bubble_r: float = 0.00045
    bubble_rise: float = 0.03

    @property
    def dt(self):
        return 1.0 / self.fps

    @property
    def dt_sub(self):
        return self.dt / self.substeps


@dataclass
class Settings:
    layout: Layout = field(default_factory=Layout)
    tube: Tube = field(default_factory=Tube)
    rack: Rack = field(default_factory=Rack)
    beaker: Beaker = field(default_factory=Beaker)
    hand: Hand = field(default_factory=Hand)
    motion: Motion = field(default_factory=Motion)
    physics: Physics = field(default_factory=Physics)
    fluid: Fluid = field(default_factory=Fluid)

    def particles_per_tube(self):
        return int(round(self.tube.fill_ml * 1e-6 / self.fluid.spacing ** 3))

    def lip_height(self):
        return self.tube.length - self.tube.grasp_y

    def plan_key(self):
        L, t, r, b, h, m = self.layout, self.tube, self.rack, self.beaker, self.hand, self.motion
        return repr([L.rack_a, L.rack_b, L.pick_slots[:L.n_tubes], L.place_slots[:L.n_tubes], L.beakers,
                     L.tube_beaker, [(q.tilt_slow, q.drain) for q in L.liquids], t.length, t.grasp_y, t.cap_top,
                     r.__dict__, b.radius, b.height, h.scale, h.bridge_cap, h.jaw_open, h.jaw_tube, h.jaw_cap,
                     h.jaw_ease, h.finger_mid, h.twist_stroke, h.twist_strokes, h.cap_clear, m.__dict__,
                     self.fluid.substeps, self.fluid.fps])


