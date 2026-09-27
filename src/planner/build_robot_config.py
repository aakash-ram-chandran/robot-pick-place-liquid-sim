from pathlib import Path

from curobo.robot_builder import RobotBuilder


class RobotConfigBuilder:
    def __init__(self, root: Path, name="iiwa14_cartesian_hand", tool="tcp"):
        self.urdf = root / "assets" / f"{name}.urdf"
        self.out = root / "configs" / f"{name}.yml"
        self.tool = tool

    def build(self):
        builder = RobotBuilder(urdf_path=str(self.urdf), asset_path=str(self.urdf.parent), tool_frames=[self.tool])
        builder.fit_collision_spheres(clip_links={"link_0": ("z", 0.0)})
        print(f"fitted {builder.num_spheres} spheres on {len(builder.collision_link_names)} links")
        builder.compute_collision_matrix()
        self.out.parent.mkdir(exist_ok=True)
        builder.save(builder.build(), str(self.out))
        print(f"wrote {self.out}")


if __name__ == "__main__":
    RobotConfigBuilder(Path(__file__).resolve().parents[2]).build()
