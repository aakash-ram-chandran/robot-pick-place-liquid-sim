import json
import threading
import time
from concurrent import futures
from pathlib import Path

import grpc
import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, JointState, Pose, ToolPoseCriteria

TOOL = "tcp"


class Planner:
    finger_links = ["hand_left_down", "hand_right_down", "hand_left_up", "hand_right_up"]

    def __init__(self, config: Path):
        cfg = MotionPlannerCfg.create(robot=str(config), collision_cache={"cuboid": 32})
        self.mp = MotionPlanner(cfg)
        print("warming up cuRobo (compiles CUDA graphs, ~30 s first time)...", flush=True)
        self.mp.warmup(enable_graph=True, num_warmup_iterations=2)
        self.names = list(self.mp.joint_names)
        self.dt = float(self.mp.trajopt_solver.config.interpolation_dt)

    def info(self, req):
        return {"joint_names": self.names, "tool": TOOL, "dt": self.dt,
                "default_q": self.mp.default_joint_state.position.flatten().cpu().tolist()}

    def set_scene(self, req):
        cuboids = [Cuboid(name=c["name"], dims=c["dims"], pose=c["pose"]) for c in req["cuboids"]]
        self.mp.update_world(Scene(cuboid=cuboids))
        return {"ok": True, "n": len(cuboids)}

    def plan_pose(self, req):
        start = torch.tensor([req["start"]], device="cuda", dtype=torch.float32)
        state = JointState.from_position(start, joint_names=self.names)
        goal = GoalToolPose.from_poses({TOOL: Pose.from_list(req["pose"])},
                                       ordered_tool_frames=self.mp.tool_frames)
        mode = req.get("mode", "free")
        if mode == "linear":
            crit = ToolPoseCriteria.linear_motion(axis=req.get("axis", "z"), project_distance_to_goal=True)
        elif mode == "hold":
            crit = ToolPoseCriteria(terminal_pose_axes_weight_factor=[1.0] * 6,
                                    non_terminal_pose_axes_weight_factor=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        else:
            crit = ToolPoseCriteria()
        self.mp.update_tool_pose_criteria({TOOL: crit})
        if req.get("ignore_fingers"):
            self.mp.disable_link_collision(self.finger_links)
        t0 = time.time()
        try:
            result = self.mp.plan_pose(goal, state, max_attempts=req.get("attempts", 5))
        finally:
            self.mp.update_tool_pose_criteria({TOOL: ToolPoseCriteria()})
            if req.get("ignore_fingers"):
                self.mp.enable_link_collision(self.finger_links)
        ok = result is not None and bool(result.success.any())
        print(f"PlanPose {mode:6s} -> {'ok' if ok else 'FAILED'} in {time.time() - t0:.2f}s", flush=True)
        if not ok:
            return {"success": False}
        q = result.get_interpolated_plan().position.reshape(-1, len(self.names))
        return {"success": True, "dt": self.dt, "q": q.cpu().tolist()}

    def plan_joints(self, req):
        start = JointState.from_position(torch.tensor([req["start"]], device="cuda", dtype=torch.float32),
                                         joint_names=self.names)
        goal = JointState.from_position(torch.tensor([req["goal"]], device="cuda", dtype=torch.float32),
                                        joint_names=self.names)
        t0 = time.time()
        result = self.mp.plan_cspace(goal, start, max_attempts=req.get("attempts", 5))
        ok = result is not None and bool(result.success.any())
        print(f"PlanJoints     -> {'ok' if ok else 'FAILED'} in {time.time() - t0:.2f}s", flush=True)
        if not ok:
            return {"success": False}
        q = result.get_interpolated_plan().position.reshape(-1, len(self.names))
        return {"success": True, "dt": self.dt, "q": q.cpu().tolist()}

    def ik_path(self, req):
        ik = self.mp.ik_solver
        n_seeds = ik.config.num_seeds
        q = torch.tensor([req["start"]], device="cuda", dtype=torch.float32)
        seeds = req.get("seeds")
        rows = []
        if req.get("ignore_fingers"):
            self.mp.disable_link_collision(self.finger_links)
        t0 = time.time()
        try:
            for i, pose in enumerate(req["poses"]):
                goal = GoalToolPose.from_poses({TOOL: Pose.from_list(pose)},
                                               ordered_tool_frames=self.mp.tool_frames)
                seed = q if seeds is None else torch.tensor([seeds[i]], device="cuda", dtype=torch.float32)
                res = ik.solve_pose(goal, current_state=JointState.from_position(q, joint_names=self.names),
                                    seed_config=seed.view(1, 1, -1).repeat(1, n_seeds, 1))
                if not bool(res.success.any()):
                    print(f"IKPath         -> FAILED at pose {i}/{len(req['poses'])}", flush=True)
                    return {"success": False, "failed_at": i}
                q = res.solution.reshape(-1, len(self.names))[:1].clone()
                rows.append(q[0].cpu().tolist())
        finally:
            if req.get("ignore_fingers"):
                self.mp.enable_link_collision(self.finger_links)
        print(f"IKPath         -> ok, {len(rows)} poses in {time.time() - t0:.2f}s", flush=True)
        return {"success": True, "q": rows}

    def fk(self, req):
        q = torch.tensor(req["q"], device="cuda", dtype=torch.float32)
        pose = self.mp.compute_kinematics(JointState.from_position(q, joint_names=self.names)) \
            .tool_poses.get_link_pose(TOOL)
        p = pose.position.reshape(-1, 3)
        quat = pose.quaternion.reshape(-1, 4)
        return {"poses": torch.cat([p, quat], dim=1).cpu().tolist()}


class PlannerServer:
    methods = {"Info": "info", "SetScene": "set_scene", "PlanPose": "plan_pose", "PlanJoints": "plan_joints",
               "IKPath": "ik_path", "FK": "fk"}

    def __init__(self, config: Path, address="127.0.0.1:50061"):
        self.planner = Planner(config)
        self.address = address
        self.lock = threading.Lock()

    def _handler(self, name):
        def call(request, context):
            req = json.loads(request)
            with self.lock:
                return json.dumps(getattr(self.planner, name)(req)).encode()
        return grpc.unary_unary_rpc_method_handler(call)

    def serve(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4),
                             options=[("grpc.max_send_message_length", 64 << 20),
                                      ("grpc.max_receive_message_length", 64 << 20)])
        server.add_generic_rpc_handlers([grpc.method_handlers_generic_handler(
            "curobo.Planner", {k: self._handler(f) for k, f in self.methods.items()})])
        server.add_insecure_port(self.address)
        server.start()
        print(f"cuRobo planner listening on {self.address}", flush=True)
        server.wait_for_termination()


if __name__ == "__main__":
    PlannerServer(Path(__file__).resolve().parents[2] / "configs" / "iiwa14_cartesian_hand.yml").serve()
