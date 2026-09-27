import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from geometry import rpy_matrix


class RobotModel:
    arm_joints = [f"joint_{i}" for i in range(7)]
    hand_slides = ["bridge", "down_jaw", "up_jaw", "down_left", "down_right", "up_left", "up_right"]
    hand_parts = [
        (["base__nylon", "base__steel", "base__red"], lambda h: (0.0, 0.0, 0.0)),
        (["bridge__nylon", "bridge__steel", "bridge__red"], lambda h: (0.0, 0.0, h[0])),
        (["left_down_rack__nylon", "left_down_rack__steel", "left_down_rack__red"], lambda h: (0.0, -h[1], 0.0)),
        (["right_down_rack__nylon", "right_down_rack__steel", "right_down_rack__red"], lambda h: (0.0, h[1], 0.0)),
        (["left_up_rack__nylon", "left_up_rack__steel", "left_up_rack__red"], lambda h: (0.0, -h[2], h[0])),
        (["right_up_rack__nylon", "right_up_rack__steel", "right_up_rack__red"], lambda h: (0.0, h[2], h[0])),
        (["left_down_finger__nylon"], lambda h: (h[3], -h[1], 0.0)),
        (["right_down_finger__nylon"], lambda h: (h[4], h[1], 0.0)),
        (["left_up_finger__nylon"], lambda h: (h[5], -h[2], h[0])),
        (["right_up_finger__nylon"], lambda h: (h[6], h[2], h[0])),
    ]

    def __init__(self, urdf: Path, hand_meshes: Path):
        self.urdf = Path(urdf)
        self.asset_dir = self.urdf.parent
        self.hand_meshes = Path(hand_meshes)
        self.joints, self.link_order, self.link_meshes = self._load(self.urdf)
        self.tcp_chain = set()
        link = "tcp"
        while link != "world":
            self.tcp_chain.add(link)
            link = self.joints[link]["parent"]

    @staticmethod
    def _load(path):
        root = ET.parse(path).getroot()
        joints = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0.0] * 3
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0.0] * 3
            joints[j.find("child").get("link")] = dict(name=j.get("name"), parent=j.find("parent").get("link"),
                                                       xyz=np.array(xyz), rpy=rpy)
        meshes = {}
        for link in root.findall("link"):
            m = link.find("visual/geometry/mesh")
            if m is not None:
                meshes[link.get("name")] = m.get("filename")
        order, todo = [], ["world"]
        while todo:
            parent = todo.pop(0)
            for child, j in joints.items():
                if j["parent"] == parent:
                    order.append(child)
                    todo.append(child)
        return joints, order, meshes

    def fk(self, q, links=None):
        q = np.atleast_2d(q)
        n = len(q)
        out = {"world": np.broadcast_to(np.eye(4), (n, 4, 4))}
        for child in self.link_order:
            if links is not None and child not in links:
                continue
            j = self.joints[child]
            T = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            T[:, :3, :3] = rpy_matrix(*j["rpy"])
            T[:, :3, 3] = j["xyz"]
            if j["name"] in self.arm_joints:
                a = q[:, self.arm_joints.index(j["name"])]
                Rz = np.zeros((n, 4, 4))
                Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1] = np.cos(a), -np.sin(a), np.sin(a), np.cos(a)
                Rz[:, 2, 2] = Rz[:, 3, 3] = 1.0
                T = T @ Rz
            out[child] = out[j["parent"]] @ T
        return out

    def fk_tcp(self, q):
        return self.fk(q, self.tcp_chain)["tcp"]

    def mesh_path(self, link):
        return self.asset_dir / self.link_meshes[link]
