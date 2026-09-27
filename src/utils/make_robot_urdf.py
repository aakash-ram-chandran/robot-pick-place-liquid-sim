import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from settings import Settings


class RobotUrdf:
    joints = [((0, 0, 0.1575), (0, 0, 0), 2.96706), ((0, 0, 0.2025), (1.570796, 0, 3.141593), 2.0944),
              ((0, 0.2045, 0), (1.570796, 0, 3.141593), 2.96706), ((0, 0, 0.2155), (1.570796, 0, 0), 2.0944),
              ((0, 0.1845, 0), (-1.570796, 3.141593, 0), 2.96706), ((0, 0, 0.2155), (1.570796, 0, 0), 2.0944),
              ((0, 0.081, 0), (-1.570796, 3.141593, 0), 3.05433)]
    inertials = [(5.0, (-0.1, 0, 0.07), (0.05, 0.06, 0.03)), (4.0, (0, -0.03, 0.12), (0.1, 0.09, 0.02)),
                 (4.0, (0.0003, 0.059, 0.042), (0.05, 0.018, 0.044)), (3.0, (0, 0.03, 0.13), (0.08, 0.075, 0.01)),
                 (2.7, (0, 0.067, 0.034), (0.03, 0.01, 0.029)), (1.7, (0.0001, 0.021, 0.076), (0.02, 0.018, 0.005)),
                 (1.8, (0, 0.0006, 0.0004), (0.025, 0.0136, 0.0247)), (0.428571, (0, 0, 0.02), (0.01, 0.01, 0.01))]
    hand_links = {
        "hand_base": (["base__nylon", "base__steel", "base__red"], 0, False),
        "hand_bridge": (["bridge__nylon", "bridge__steel", "bridge__red"], 0, True),
        "hand_left_down": (["left_down_rack__nylon", "left_down_rack__steel", "left_down_rack__red",
                            "left_down_finger__nylon"], -1, False),
        "hand_right_down": (["right_down_rack__nylon", "right_down_rack__steel", "right_down_rack__red",
                             "right_down_finger__nylon"], 1, False),
        "hand_left_up": (["left_up_rack__nylon", "left_up_rack__steel", "left_up_rack__red",
                          "left_up_finger__nylon"], -1, True),
        "hand_right_up": (["right_up_rack__nylon", "right_up_rack__steel", "right_up_rack__red",
                           "right_up_finger__nylon"], 1, True),
    }

    def __init__(self, assets: Path, settings=None, flange_z=0.045, adapter=0.010, post_len=0.12, post_r=0.018,
                 plate_t=0.006, hand_back=-0.047, hand_bottom=-0.016, tcp_x=0.105):
        s = settings or Settings()
        self.assets = assets
        self.scale = s.hand.scale
        self.jaw = s.hand.jaw_open
        self.bridge = s.hand.bridge_cap
        self.flange_z = flange_z
        self.adapter = adapter
        self.post_len = post_len
        self.post_r = post_r
        self.plate_t = plate_t
        self.hand_back = hand_back
        self.hand_bottom = hand_bottom
        self.tcp_x = tcp_x

    def _post_obj(self, path, n=32):
        th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        r = self.post_r + 0.002
        v = [(r * np.cos(a), r * np.sin(a), z) for z in (0.0, self.post_len) for a in th] + \
            [(0, 0, 0), (0, 0, self.post_len)]
        f = []
        for i in range(n):
            j = (i + 1) % n
            f += [(i, j, n + j), (i, n + j, n + i), (2 * n, j, i), (2 * n + 1, n + i, n + j)]
        path.write_text("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in v)
                        + "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f))

    def _arm(self):
        out = '  <link name="world"/>\n  <joint name="world_joint" type="fixed">\n    <parent link="world"/>\n' \
              '    <child link="link_0"/>\n    <origin xyz="0 0 0" rpy="0 0 0"/>\n  </joint>\n'
        for i, (m, com, (ixx, iyy, izz)) in enumerate(self.inertials):
            out += f'''  <link name="link_{i}">
    <inertial>
      <origin xyz="{com[0]} {com[1]} {com[2]}" rpy="0 0 0"/>
      <mass value="{m}"/>
      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
    </inertial>
    <visual><geometry><mesh filename="iiwa14/meshes/visual/link_{i}.stl"/></geometry></visual>
    <collision><geometry><mesh filename="iiwa14/meshes/collision/link_{i}.stl"/></geometry></collision>
  </link>
'''
            if i < 7:
                xyz, rpy, lim = self.joints[i]
                out += f'''  <joint name="joint_{i}" type="revolute">
    <parent link="link_{i}"/>
    <child link="link_{i + 1}"/>
    <origin xyz="{xyz[0]} {xyz[1]} {xyz[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>
    <axis xyz="0 0 1"/>
    <limit effort="300" lower="{-lim}" upper="{lim}" velocity="10"/>
  </joint>
'''
        return out

    def _tool(self):
        R = np.diag([1.0, -1.0, -1.0])
        end = np.array([0.0, 0.0, self.flange_z + self.adapter + self.post_len])
        corner = self.scale * np.array([self.hand_back, 0.0, self.hand_bottom]) - [self.plate_t + self.post_r, 0.0, 0.0]
        p = end - R @ corner
        k = 0.001 * self.scale
        out = f'''  <joint name="extension_joint" type="fixed">
    <parent link="link_7"/>
    <child link="extension"/>
    <origin xyz="0 0 {self.flange_z + self.adapter:.6f}" rpy="0 0 0"/>
  </joint>
  <link name="extension">
    <visual><geometry><mesh filename="cartesian_hand/meshes/extension.obj"/></geometry></visual>
    <collision><geometry><mesh filename="cartesian_hand/meshes/extension.obj"/></geometry></collision>
  </link>
  <joint name="hand_mount" type="fixed">
    <parent link="link_7"/>
    <child link="hand"/>
    <origin xyz="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" rpy="3.141593 0 0"/>
  </joint>
  <link name="hand"/>
'''
        for link, (meshes, side, up) in self.hand_links.items():
            vis = "".join(f'    <visual><geometry><mesh filename="cartesian_hand/meshes/{m}.obj" '
                          f'scale="{k} {k} {k}"/></geometry></visual>\n' for m in meshes)
            out += f'''  <joint name="{link}_joint" type="fixed">
    <parent link="hand"/>
    <child link="{link}"/>
    <origin xyz="0 {side * self.jaw} {self.bridge if up else 0}" rpy="0 0 0"/>
  </joint>
  <link name="{link}">
{vis}    <collision><geometry><mesh filename="cartesian_hand/meshes/{meshes[0]}.obj" scale="{k} {k} {k}"/></geometry></collision>
  </link>
'''
        out += f'''  <joint name="tcp_joint" type="fixed">
    <parent link="hand"/>
    <child link="tcp"/>
    <origin xyz="{self.tcp_x * self.scale:.6f} 0 0" rpy="1.570796 0 1.570796"/>
  </joint>
  <link name="tcp"/>
'''
        return out

    def write(self):
        self._post_obj(self.assets / "cartesian_hand" / "meshes" / "extension.obj")
        path = self.assets / "iiwa14_cartesian_hand.urdf"
        path.write_text('<?xml version="1.0"?>\n<robot name="iiwa14_cartesian_hand">\n'
                        + self._arm() + self._tool() + "</robot>\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    RobotUrdf(Path(__file__).resolve().parents[2] / "assets").write()
