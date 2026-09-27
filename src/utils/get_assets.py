import shutil
import subprocess
import tempfile
from pathlib import Path

from make_robot_urdf import RobotUrdf


class AssetFetcher:
    sources = {
        "iiwa_stack": ("https://github.com/IFL-CAMP/iiwa_stack", "iiwa_description/meshes/iiwa14", "iiwa14/meshes"),
        "Cartesian_Hand": ("https://github.com/generalroboticslab/Cartesian_Hand", "assets/cartesian_hand/meshes",
                           "cartesian_hand/meshes"),
    }

    def __init__(self, assets: Path):
        self.assets = assets

    def fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, (url, src, dst) in self.sources.items():
                repo = Path(tmp) / name
                print(f"cloning {url}")
                subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(repo)],
                               check=True)
                subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", src], check=True)
                out = self.assets / dst
                if out.exists():
                    shutil.rmtree(out)
                shutil.copytree(repo / src, out, ignore=shutil.ignore_patterns("collision_pieces"))
                print(f"  -> {out}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    AssetFetcher(root / "assets").fetch()
    RobotUrdf(root / "assets").write()
