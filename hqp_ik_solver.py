#!/usr/bin/env python3
"""
改进的HQP IK求解器
- 更好的数值稳定性
- 自适应扰动策略
- 性能优化
- 错误处理机制
- 参数自适应调整
"""
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
import meshcat_shapes
import os.path as osp
import time
import cvxpy as cp
from scipy.linalg import svd, qr
import threading
import warnings
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
from scipy.linalg import null_space
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Header
except Exception:
    rclpy = None
    Node = object
    JointState = None
    Header = None
 
 
class NullspaceSinePerturbation:
    """
    单一正弦零空间扰动 + 关节限位衰减
    - 幅度不超过2
    - 在 q_mid 最大，接近 q_min / q_max 时平滑衰减为 0
    - 周期正弦扰动，保证运动自然
    """
    def __init__(self, null_dim: int, dt: float = 0.02,
                 base_period: float = 5.0, amp_scale: float = 2.0,
                 q_min: np.ndarray = None, q_max: np.ndarray = None):
        self.null_dim = null_dim
        self.dt = dt
        self.t = 0.0
        self.base_period = base_period
        self.amp_scale = amp_scale
        self.q_min = q_min
        self.q_max = q_max
        if q_min is not None and q_max is not None:
            self.q_mid = 0.5 * (q_min + q_max)

    def _limit_envelope(self, q: np.ndarray) -> np.ndarray:
        """
        平滑关节限位 envelope:
        - 在 q_mid = 1
        - 在接近 q_min/q_max 时 → 0
        - 使用三次平滑函数，保证一阶连续
        """
        # 将 q 映射到 [-1,1]，q_mid -> 0
        normalized_centered = np.tanh(2*(q.flatten()-self.q_mid)/(self.q_max-self.q_min) + 1e-8)
        # 三次平滑 envelope: q_mid=0 -> 1, q_min/q_max -> 0
        envelope = 1 - 3 * normalized_centered**2 + 2 * normalized_centered**3

        return envelope

    def step(self, q: np.ndarray) -> np.ndarray:
        """生成零空间扰动向量"""
        self.t += self.dt
        omega = 2 * np.pi / self.base_period

        # 基本正弦波
        sine_val = np.sin(omega * self.t)

        # 关节限位 envelope
        envelope = self._limit_envelope(q)

        # 每个自由度扰动
        z_ref = self.amp_scale * sine_val * envelope[:self.null_dim]
        return z_ref

class ROS2JointStatePublisher(Node):
    """ROS2关节状态发布器：直接发布接收到的(q, dq)到/right/ik_robstride_joint_cmd"""
    def __init__(self, node_name: str = "hqp_ik_joint_publisher_right"):
        super().__init__(node_name)
        self.pub = self.create_publisher(JointState, '/right/ik_robstride_joint_cmd', 10)

    def publish_now(self, joint_names, q, dq=None):
        if JointState is None:
            return
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(joint_names)
        msg.position = np.asarray(q, dtype=float).tolist()
        if dq is None:
            msg.velocity = [0.0] * len(msg.name)
        else:
            msg.velocity = np.asarray(dq, dtype=float).tolist()
        msg.effort = [0.0] * len(msg.name)
        self.pub.publish(msg)
    

class HQPController:
    """HQP机械臂控制器"""
    
    def __init__(self, urdf_path: str, target_frame_name: str, 
                 dt: float = 0.02, visualize: bool = True,
                 ros2_node: 'ROS2JointStatePublisher' = None):
        # 机器人模型
        self.robot = RobotWrapper.BuildFromURDF(
            urdf_path, package_dirs=[osp.dirname(urdf_path)]
        )
        self.FRAME_ID = self.robot.model.getFrameId(target_frame_name)
        self.dt = dt

        # 可选ROS2节点
        self.ros2_node = ros2_node
        
        # 关节状态（保持原有初始化）
        self.q = pin.neutral(self.robot.model)
        self.dq = np.zeros_like(self.q)

        # 关节限位
        self.q_min = self.robot.model.lowerPositionLimit.copy()
        self.q_max = self.robot.model.upperPositionLimit.copy()
        self.q_mid = 0.5 * (self.q_min + self.q_max)
        self.nq = self.q_min.shape[0]
        # 关节名称（用于ROS2发布）
        self.joint_names = [self.robot.model.names[i] for i in range(1, self.nq + 1)]
        
        # 打印初始配置信息
        print(f"初始关节配置: {self.q}")
        print(f"关节限位: min={self.q_min}, max={self.q_max}")
        print(f"目标帧ID: {self.FRAME_ID}")
        
        # 速度限位
        self.dq_max = self.robot.model.velocityLimit.copy()
        if self.dq_max is None:
            self.dq_max = np.full(self.nq, 0.5)  # 默认值
        
        # 控制器参数（保持原有值)
        self.Kp_task = 1.0
        self.alpha_limit = 10.0
        self.beta_perturb = 0.3
        self.switch_err_threshold = 1e-3
        # 新增：任务空间速度上限与关节加速度上限（提升平滑性）
        self.v_task_max = 0.5  # 任务空间速度范数上限（m/s 与 rad/s 混合量纲）

        self.prev_dq = np.zeros_like(self.q)      # 上一步的关节速度（用于平滑）
        # 平滑权重（可调）
        self.w_vel_smooth = 1e-2    # 惩罚 dq 与上一步 dq 突变，越大越平滑但响应慢
        self.w_jerk = 1e-3          # 惩罚加速度（近似 jerk），按需打开
 
        # 自适应参数
        self.performance_window = 50
        self.task_error_history = []
        self.solve_time_history = []
 
        # 零空间扰动
        self.null_perturb = None
        self.hqp_enabled = False
        
        # 可视化
        if visualize:
            self.viewer = MeshcatVisualizer(
                self.robot.model, self.robot.collision_model, self.robot.visual_model
            )
            self.viewer.initViewer(open=True)
            self.viewer.loadViewerModel()
            self.viewer.display(self.q)
        
        # 日志记录（用于性能分析）
        self.control_time = 0.0
        self.log_t = []
        self.log_q = []
        self.log_dq = []
        self.log_error = []
        self.log_solve_time = []
        self.log_status = []
        self.log_nullspace_usage = []
        self.log_joint_velocity_norm = []
        self.log_joint_limit_violation = []
        
        # 若提供ROS2节点，发布初始状态
        if self.ros2_node is not None:
            try:
                self.ros2_node.publish_now(self.joint_names, self.q, self.dq)
                print("初始关节状态已发布到: /right/ik_robstride_joint_cmd")
            except Exception as e:
                print(f"ROS2初始发布失败: {e}")
    
    def _compute_jacobian_robust(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """鲁棒的雅可比矩阵计算"""
        try:
            # 计算雅可比矩阵
            J = pin.computeFrameJacobian(self.robot.model, self.robot.data, q, self.FRAME_ID)
            
            # 使用SVD进行数值稳定的秩估计和零空间计算
            U, s, Vh = svd(J, full_matrices=True)
            
            # 基于奇异值估计秩
            tol = 1e-6
            rank = int(np.sum(s > tol))
            
            # 用scipy的null_space计算零空间基，更数值稳健
            null_basis = null_space(J, rcond=1e-6) 

            return J, null_basis, rank
            
        except Exception as e:
            warnings.warn(f"Jacobian computation failed: {e}")
            # 返回单位矩阵作为备选
            return np.eye(6, self.nq), np.empty((self.nq, 0)), 6
    
    def _compute_task_error(self, goal_pose: pin.SE3) -> Tuple[np.ndarray, float]:
        """计算任务误差（完全按照原始代码）"""
        oMf = self.robot.data.oMf[self.FRAME_ID]
        err = pin.log(goal_pose.actInv(oMf)).vector
        err_norm = np.linalg.norm(err)
        return err, err_norm

    def _map_cvx_status(self, status: str) -> str:
        """将cvxpy状态映射为标准求解状态字符串"""
        if status in ["optimal", "optimal_inaccurate"]:
            return "success"
        if status == "infeasible":
            return "infeasible"
        if status == "unbounded":
            return "unbounded"
        return "solver_error"
    
    def _update_performance_metrics(self, task_error: float, solve_time: float):
        """更新性能指标"""
        self.task_error_history.append(task_error)
        self.solve_time_history.append(solve_time)
        
        if len(self.task_error_history) > self.performance_window:
            self.task_error_history.pop(0)
        if len(self.solve_time_history) > self.performance_window:
            self.solve_time_history.pop(0)
    
    def _compute_joint_limits_violation(self, q: np.ndarray) -> float:
        """计算关节限位违反程度"""
        violation = 0.0
        for i in range(self.nq):
            if q[i] < self.q_min[i]:
                violation += (self.q_min[i] - q[i]) ** 2
            elif q[i] > self.q_max[i]:
                violation += (q[i] - self.q_max[i]) ** 2
        return np.sqrt(violation)

    def _compute_adaptive_task_weights(self, err: np.ndarray) -> np.ndarray:
        """
        自适应任务权重（6 自由度逐项分配 + 能量感知 + 平滑归一化）
        """
        err_abs = np.abs(err)
        err_sum = np.sum(err_abs) + 1e-8

        # 基础能量代价 (可根据机器人动力学调整)
        energy_cost = np.array([1.2, 1.2, 1.5,   # 位置 (z方向贵)
                                1.3, 1.3, 0.8])  # 姿态 (yaw最便宜)
        beta = 1.5   # 能量代价平滑指数

        # 归一化误差
        err_normed = err_abs / err_sum

        # softmax平滑化误差
        soft_err = np.exp(err_normed)
        soft_err /= np.sum(soft_err + 1e-8)

        # 最终权重
        weights = 1.0 + 3.0 * (soft_err / (energy_cost ** beta))

        return np.diag(weights)

    def step(self, goal_pose: pin.SE3) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """执行一个控制步（重构版，支持整体速度平滑）"""
        start_time = time.time()

        # 前向运动学
        pin.forwardKinematics(self.robot.model, self.robot.data, self.q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)

        # 任务误差
        err, err_norm = self._compute_task_error(goal_pose)
        v_task = -self.Kp_task * err

        # 任务速度上限
        v_norm = float(np.linalg.norm(v_task))
        if v_norm > self.v_task_max and v_norm > 1e-12:
            v_task *= self.v_task_max / v_norm

        # 调试信息
        oMf = self.robot.data.oMf[self.FRAME_ID]
        print(f"当前位姿: pos={oMf.translation}, 目标位姿: pos={goal_pose.translation}")
        print(f"误差向量: {err}")
        print(f"误差: {err_norm:.6f}, 阈值: {self.switch_err_threshold}")
        print(f"任务速度: {v_task}")

        # 雅可比矩阵
        J, null_basis, rank = self._compute_jacobian_robust(self.q)

        # 关节速度约束
        dq_min = np.maximum((self.q_min - self.q.flatten()) / self.dt, -self.dq_max)
        dq_max = np.minimum((self.q_max - self.q.flatten()) / self.dt, self.dq_max)

        # 一级QP
        dq_var = cp.Variable(self.nq)
        W_task = self._compute_adaptive_task_weights(err)
        obj_task = cp.sum_squares(np.sqrt(W_task) @ (J @ dq_var - v_task))
        s_vals = np.linalg.svd(J, compute_uv=False)
        sigma_min = float(np.min(s_vals)) if s_vals.size > 0 else 0.0
        lambda_reg = 1e-3 if sigma_min > 1e-1 else 1e-1
        obj_reg = lambda_reg * cp.sum_squares(dq_var)

        obj_primary = obj_task + obj_reg

        constraints = [dq_var >= dq_min, dq_var <= dq_max]

        prob1 = cp.Problem(cp.Minimize(obj_primary), constraints)
        prob1.solve(solver=cp.OSQP, warm_start=True, verbose=False, eps_abs=1e-4, eps_rel=1e-4, max_iter=20000)
        status_primary = self._map_cvx_status(prob1.status)
        dq1 = dq_var.value.reshape((self.nq, 1))

        # 启用二级HQP
        if not self.hqp_enabled and err_norm < self.switch_err_threshold:
            self.hqp_enabled = True
            null_dim = self.nq - rank
            if null_dim > 0:
                self.null_perturb = NullspaceSinePerturbation(null_dim=null_dim, dt=self.dt, amp_scale=1.5,
                                                            q_min=self.q_min, q_max=self.q_max)
                print(">>> HQP 二级扰动已启用")

        # 二级QP
        dq_total = dq1.copy()
        if self.hqp_enabled and null_basis.size > 0:
            null_dim = self.nq - rank
            z_var = cp.Variable(null_dim)
            dq_expr = dq1.flatten() + null_basis @ z_var

            normalized = 2.0 * ((self.q.flatten() + dq_expr * self.dt) - self.q_mid) / ((self.q_max - self.q_min) + 1e-8)
            obj_limits = cp.sum_squares(normalized)
            z_ref = self.null_perturb.step(self.q) if self.null_perturb else np.zeros(null_dim)
            obj_perturb = cp.sum_squares(z_var - z_ref)

            # 整体速度平滑 & jerk
            obj_vel_smooth = self.w_vel_smooth * cp.sum_squares(dq_expr - self.prev_dq)
            obj_jerk = self.w_jerk * cp.sum_squares((dq_expr - self.prev_dq) / max(self.dt, 1e-6))

            obj_secondary = self.alpha_limit * obj_limits + self.beta_perturb * obj_perturb + obj_vel_smooth + obj_jerk

            constraints2 = [dq_expr >= dq_min, dq_expr <= dq_max, z_var >= -2.0, z_var <= 2.0]
            prob2 = cp.Problem(cp.Minimize(obj_secondary), constraints2)
            prob2.solve(solver=cp.OSQP, warm_start=True, verbose=False, eps_abs=1e-4, eps_rel=1e-4, max_iter=20000)

            if prob2.status in ["optimal", "optimal_inaccurate"]:
                dq_total = (dq1 + (null_basis @ z_var.value).reshape((self.nq, 1)))
            else:
                dq_total = dq1
            if err_norm > 30 * self.switch_err_threshold:
                dq_total = np.zeros_like(dq_total)
                print(">>> 零空间运动严重影响了末端执行器位姿不变")

        # 限幅
        max_step = 1.0
        norm_dq = np.linalg.norm(dq_total)
        if norm_dq > max_step:
            dq_total = dq_total / norm_dq * max_step

        # 更新关节
        self.q = pin.integrate(self.robot.model, self.q, dq_total.flatten() * self.dt)
        self.q = pin.normalize(self.robot.model, self.q)
        self.dq = dq_total.flatten()
        self.prev_dq = self.dq.copy()

        # 更新性能指标 & 日志
        solve_time = time.time() - start_time
        self._update_performance_metrics(err_norm, solve_time)
        joint_violation = self._compute_joint_limits_violation(self.q)
        combined_status = status_primary
        if self.hqp_enabled:
            combined_status = "success" if prob2.status in ["optimal", "optimal_inaccurate"] else status_primary

        self.control_time += self.dt
        self.log_t.append(self.control_time)
        self.log_q.append(self.q.copy())
        self.log_dq.append(self.dq.copy())
        self.log_error.append(float(err_norm))
        self.log_solve_time.append(float(solve_time))
        self.log_status.append(combined_status)
        self.log_nullspace_usage.append(bool(self.hqp_enabled and (self.nq - rank) > 0))
        self.log_joint_velocity_norm.append(float(np.linalg.norm(self.dq)))
        self.log_joint_limit_violation.append(float(joint_violation))

        # 返回
        result_info = {
            'error_norm': err_norm,
            'hqp_enabled': self.hqp_enabled,
            'dq_norm': float(np.linalg.norm(self.dq)),
            'solve_time': solve_time,
            'rank': rank,
            'null_dim': self.nq - rank
        }

        return self.q, self.dq, result_info


    def get_performance_data(self) -> Dict[str, Any]:
        """导出完整性能数据（供分析器使用）"""
        return {
            'solve_times': list(self.log_solve_time),
            'task_errors': list(self.log_error),
            'solver_statuses': list(self.log_status),
            'nullspace_usage': list(self.log_nullspace_usage),
            'joint_velocities': list(self.log_joint_velocity_norm),
            'joint_limit_violations': list(self.log_joint_limit_violation),
            'timestamps': list(self.log_t),
            'q_samples': np.vstack(self.log_q).tolist() if self.log_q else [],
            'dq_samples': np.vstack(self.log_dq).tolist() if self.log_dq else []
        }

    def run_control_loop(self, goal_pose: pin.SE3, runtime: float = 20.0):
        """运行控制循环"""
        t0 = time.time()
        step = 0
        
        print(f"开始控制循环，目标运行时间: {runtime}秒")
        
        while time.time() - t0 < runtime:
            q, dq, info = self.step(goal_pose)
            
            # 可视化
            if hasattr(self, 'viewer'):
                self.viewer.display(q)
            if self.ros2_node is not None:
                try:
                    self.ros2_node.publish_now(self.joint_names, q, dq)
                except Exception as e:
                    print(f"ROS2发布失败: {e}")
            # 打印状态
            if step % 50 == 0:
                print(f"[{step:4d}] "
                      f"err={info['error_norm']:.4e} "
                      f"HQP={info['hqp_enabled']} "
                      f"||dq||={info['dq_norm']:.4e} "
                      f"rank={info['rank']} "
                      f"null_dim={info['null_dim']} "
                      f"time={info['solve_time']*1000:.1f}ms")
            
            step += 1
            time.sleep(self.dt)
        
        print(f"控制循环结束，总步数: {step}")


def main():
    """主函数"""
    # 初始化ROS2（如果可用）
    ros2_initialized = False
    ros2_node = None
    
    if rclpy is not None:
        try:
            rclpy.init()
            ros2_node = ROS2JointStatePublisher()
            ros2_initialized = True
            print("ROS2初始化成功")
        except Exception as e:
            print(f"ROS2初始化失败: {e}")
    else:
        print("rclpy不可用，将在非ROS2模式下运行")
    
    # 机器人模型路径
    urdf_path = osp.join(
        osp.dirname(__file__),
        "7dof_robstride",
        "robstride_right.urdf",
    )
    
    # 创建控制器（传入ROS2节点）
    arm = HQPController(urdf_path, target_frame_name="r_joint7", dt=0.01, visualize=True, ros2_node=ros2_node)
    
    # 设置目标位姿
    desired_rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    goal_pose = pin.SE3(desired_rot, np.array([0.5, 0.0, 0.0]))
    print(f"目标位姿设置: 旋转矩阵=\n{desired_rot}")
    print(f"目标位置: {goal_pose.translation}")
    
    # 可视化目标
    if hasattr(arm, 'viewer'):
        meshcat_shapes.frame(arm.viewer.viewer["target"], opacity=0.5)
        arm.viewer.viewer["target"].set_transform(goal_pose.np)
    
    try:
        # 运行控制循环
        arm.run_control_loop(goal_pose, runtime=200.0)
        
    finally:
        # 清理ROS2资源
        if ros2_initialized and ros2_node is not None:
            ros2_node.destroy_node()
            rclpy.shutdown()
            print("ROS2资源已清理")


if __name__ == "__main__":
    main()


