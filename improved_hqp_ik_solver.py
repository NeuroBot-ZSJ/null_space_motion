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
from scipy.interpolate import CubicSpline
from scipy.linalg import svd, qr


def soft_square_wave(t, period=8.0, sharpness=6.0):
    """生成平滑方波，sharpness越大越接近方波（保持原有函数）"""
    phase = (t % period) / period
    return np.tanh(np.sin(phase * 2 * np.pi) * sharpness)
import warnings
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum


class SolverStatus(Enum):
    """求解器状态枚举"""
    SUCCESS = "success"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    SOLVER_ERROR = "solver_error"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class SolverResult:
    """求解结果数据类"""
    status: SolverStatus
    solution: Optional[np.ndarray]
    objective_value: Optional[float]
    solve_time: float
    iterations: int
    message: str


class AdaptiveNullspacePerturbation:
    """自适应零空间扰动策略（保持原有平滑方波特性）"""
    
    def __init__(self, null_dim: int, dt: float = 0.02, 
                 base_period: float = 16.0, num_control_points: int = 8, amp_scale: float = 5.0):
        self.null_dim = null_dim
        self.dt = dt
        self.t = 0.0
        self.base_period = base_period
        self.amp_scale = amp_scale
        
        # 样条控制点（每个自由度一条曲线）- 保持原有逻辑
        control_x = np.linspace(0, base_period, num_control_points, endpoint=False)
        control_y = (np.random.rand(null_dim, num_control_points) * 2 - 1) * amp_scale
        control_x = np.append(control_x, base_period)
        control_y = np.hstack([control_y, control_y[:, [0]]])  # 封闭周期
        
        self.splines = []
        for i in range(null_dim):
            self.splines.append(CubicSpline(control_x, control_y[i, :], bc_type='periodic'))
        
        # 自适应参数
        self.performance_history = []
        self.max_history = 100
        
    def step(self, task_error: float = None, joint_limits_violation: float = None) -> np.ndarray:
        """生成自适应扰动（保持原有平滑方波特性）"""
        self.t += self.dt
        t_mod = self.t % self.base_period
        
        # 样条基值（保持原有逻辑）
        values = np.array([spline(t_mod) for spline in self.splines])
        
        # 根据任务性能自适应调整幅度（如果提供了参数）
        if task_error is not None and len(self.performance_history) > 0:
            avg_performance = np.mean(self.performance_history)
            if task_error > avg_performance * 1.2:  # 任务误差增大
                values *= 0.5  # 减少扰动
            elif task_error < avg_performance * 0.8:  # 任务误差减小
                values *= 1.2  # 增加扰动
        
        # 根据关节限位违反程度调整（如果提供了参数）
        if joint_limits_violation is not None and joint_limits_violation > 0.1:
            values *= 0.3  # 大幅减少扰动
        
        # 更新性能历史（如果提供了参数）
        if task_error is not None:
            self.performance_history.append(task_error)
            if len(self.performance_history) > self.max_history:
                self.performance_history.pop(0)
        
        # 保持原有的幅度缩放
        return values * 0.8


class RobustHQPSolver:
    """鲁棒的HQP求解器"""
    
    def __init__(self, n_vars: int, n_constraints: int):
        self.n_vars = n_vars
        self.n_constraints = n_constraints
        self.solver_options = {
            'solver': cp.OSQP,
            'warm_start': True,
            'verbose': False,
            'eps_abs': 1e-6,
            'eps_rel': 1e-6,
            'max_iter': 10000
        }
        
    def solve(self, P: np.ndarray, q: np.ndarray, 
              A: np.ndarray, b: np.ndarray,
              lb: np.ndarray, ub: np.ndarray) -> SolverResult:
        """求解QP问题"""
        start_time = time.time()
        
        try:
            # 创建变量
            x = cp.Variable(self.n_vars)
            
            # 构建问题
            objective = cp.quad_form(x, P) + q.T @ x
            constraints = [A @ x <= b, lb <= x, x <= ub]
            
            problem = cp.Problem(cp.Minimize(objective), constraints)
            
            # 求解
            problem.solve(**self.solver_options)
            
            solve_time = time.time() - start_time
            
            if problem.status == "optimal":
                return SolverResult(
                    status=SolverStatus.SUCCESS,
                    solution=x.value,
                    objective_value=problem.value,
                    solve_time=solve_time,
                    iterations=0,  # OSQP不直接提供迭代次数
                    message="Optimal solution found"
                )
            elif problem.status == "infeasible":
                return SolverResult(
                    status=SolverStatus.INFEASIBLE,
                    solution=None,
                    objective_value=None,
                    solve_time=solve_time,
                    iterations=0,
                    message="Problem is infeasible"
                )
            elif problem.status == "unbounded":
                return SolverResult(
                    status=SolverStatus.UNBOUNDED,
                    solution=None,
                    objective_value=None,
                    solve_time=solve_time,
                    iterations=0,
                    message="Problem is unbounded"
                )
            else:
                return SolverResult(
                    status=SolverStatus.SOLVER_ERROR,
                    solution=None,
                    objective_value=None,
                    solve_time=solve_time,
                    iterations=0,
                    message=f"Solver error: {problem.status}"
                )
                
        except Exception as e:
            solve_time = time.time() - start_time
            return SolverResult(
                status=SolverStatus.SOLVER_ERROR,
                solution=None,
                objective_value=None,
                solve_time=solve_time,
                iterations=0,
                message=f"Exception: {str(e)}"
            )


class ImprovedHQPArm:
    """改进的HQP机械臂控制器"""
    
    def __init__(self, urdf_path: str, target_frame_name: str, 
                 dt: float = 0.02, visualize: bool = True):
        # 机器人模型
        self.robot = RobotWrapper.BuildFromURDF(
            urdf_path, package_dirs=[osp.dirname(urdf_path)]
        )
        self.FRAME_ID = self.robot.model.getFrameId(target_frame_name)
        self.dt = dt
        
        # 关节状态（保持原有初始化）
        self.q = pin.neutral(self.robot.model)
        self.dq = np.zeros_like(self.q)
        
        # 关节限位
        self.q_min = self.robot.model.lowerPositionLimit.copy()
        self.q_max = self.robot.model.upperPositionLimit.copy()
        self.q_mid = 0.5 * (self.q_min + self.q_max)
        self.nq = self.q_min.shape[0]
        
        # 打印初始配置信息
        print(f"初始关节配置: {self.q}")
        print(f"关节限位: min={self.q_min}, max={self.q_max}")
        print(f"目标帧ID: {self.FRAME_ID}")
        
        # 速度限位
        self.dq_max = self.robot.model.velocityLimit.copy()
        if self.dq_max is None:
            self.dq_max = np.full(self.nq, 2.0) # 默认值
        
        # 控制器参数（保持原有值)
        self.Kp_task = 1.6
        self.alpha_limit = 3.0
        self.beta_perturb = 0.8
        self.switch_err_threshold = 1e-3
 
        # 自适应参数
        self.performance_window = 50
        self.task_error_history = []
        self.solve_time_history = []
 
        # 零空间扰动
        self.null_perturb = None
        self.hqp_enabled = False
        # 扰动模式：'fourier' 或 'spline'
        self.perturbation_mode = 'fourier'

        # HQP求解器
        self.hqp_solver = RobustHQPSolver(self.nq, 2 * self.nq)
        
        # 可视化
        if visualize:
            self.viewer = MeshcatVisualizer(
                self.robot.model, self.robot.collision_model, self.robot.visual_model
            )
            self.viewer.initViewer(open=True)
            self.viewer.loadViewerModel()
            self.viewer.display(self.q)
    
    def _compute_jacobian_robust(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """鲁棒的雅可比矩阵计算"""
        try:
            # 计算雅可比矩阵
            J = pin.computeFrameJacobian(self.robot.model, self.robot.data, q, self.FRAME_ID)
            
            # 使用SVD进行数值稳定的秩估计和零空间计算
            U, s, Vh = svd(J, full_matrices=True)
            
            # 基于奇异值估计秩
            tol = 1e-8
            rank = int(np.sum(s > tol))
            
            # 计算零空间基
            if rank < self.nq:
                null_basis = Vh[rank:, :].T
            else:
                null_basis = np.empty((self.nq, 0))
            
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
    
    def _update_performance_metrics(self, task_error: float, solve_time: float):
        """更新性能指标"""
        self.task_error_history.append(task_error)
        self.solve_time_history.append(solve_time)
        
        if len(self.task_error_history) > self.performance_window:
            self.task_error_history.pop(0)
        if len(self.solve_time_history) > self.performance_window:
            self.solve_time_history.pop(0)
    
    def _adaptive_gains(self) -> Tuple[float, float]:
        """自适应增益调整"""
        if len(self.task_error_history) < 10:
            return self.Kp_task, self.alpha_limit
        
        # 基于任务误差趋势调整增益
        recent_errors = self.task_error_history[-10:]
        error_trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]
        
        # 如果误差在增加，增加增益
        if error_trend > 0:
            kp_scale = 1.2
            alpha_scale = 1.1
        # 如果误差在减少，适当减少增益
        elif error_trend < -0.01:
            kp_scale = 0.9
            alpha_scale = 0.95
        else:
            kp_scale = 1.0
            alpha_scale = 1.0
        
        return self.Kp_task * kp_scale, self.alpha_limit * alpha_scale
    
    def _compute_joint_limits_violation(self, q: np.ndarray) -> float:
        """计算关节限位违反程度"""
        violation = 0.0
        for i in range(self.nq):
            if q[i] < self.q_min[i]:
                violation += (self.q_min[i] - q[i]) ** 2
            elif q[i] > self.q_max[i]:
                violation += (q[i] - self.q_max[i]) ** 2
        return np.sqrt(violation)
    
    def _compute_damped_pseudoinverse(self, J: np.ndarray, min_sigma: float = 1e-3) -> np.ndarray:
        """阻尼最小二乘伪逆，基于奇异值自适应正则化，提高奇异点鲁棒性"""
        U, s, Vh = svd(J, full_matrices=False)
        # 根据最小奇异值设置阻尼系数（条件数越差，阻尼越大）
        sigma_min = np.max([np.min(s), 1e-9])
        lambda_reg = (min_sigma / sigma_min) ** 2
        s_damped = s / (s**2 + lambda_reg)
        J_pinv = Vh.T @ np.diag(s_damped) @ U.T
        return J_pinv

    def _compute_adaptive_task_weights(self, err: np.ndarray) -> np.ndarray:
        """根据任务误差自适应调整位置与姿态的权重（位置误差大→位置权重点高）"""
        pos_err = np.linalg.norm(err[:3])
        ori_err = np.linalg.norm(err[3:])
        pos_w = 1.0 + 2.0 * (pos_err / (pos_err + ori_err + 1e-8))
        ori_w = 1.0 + 2.0 * (ori_err / (pos_err + ori_err + 1e-8))
        W = np.diag([pos_w, pos_w, pos_w, ori_w, ori_w, ori_w])
        return W
    
    def step(self, goal_pose: pin.SE3) -> Tuple[np.ndarray, Dict[str, Any]]:
        """执行一个控制步"""
        start_time = time.time()
        
        # 前向运动学
        pin.forwardKinematics(self.robot.model, self.robot.data, self.q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        
        # 计算任务误差（完全按照原始代码）
        err, err_norm = self._compute_task_error(goal_pose)
        
        # 任务速度（保持原有逻辑）
        v_task = -self.Kp_task * err
        
        # 调试信息
        oMf = self.robot.data.oMf[self.FRAME_ID]
        print(f"当前位姿: pos={oMf.translation}, 目标位姿: pos={goal_pose.translation}")
        print(f"误差向量: {err}")
        print(f"误差: {err_norm:.6f}, 阈值: {self.switch_err_threshold}")
        print(f"任务速度: {v_task}")
        
        # 计算雅可比矩阵
        J, null_basis, rank = self._compute_jacobian_robust(self.q)
        
        # 关节速度限位
        dq_min = (self.q_min - self.q.flatten()) / self.dt
        dq_max = (self.q_max - self.q.flatten()) / self.dt
        dq_min = np.maximum(dq_min, -self.dq_max)
        dq_max = np.minimum(dq_max, self.dq_max)
        
        # 一级QP：任务优先级（加入自适应权重与阻尼正则化）
        # 自适应任务权重
        W_task = self._compute_adaptive_task_weights(err)
        W_sqrt = np.sqrt(W_task)
        # 自适应正则化（依据最小奇异值）
        s_vals = np.linalg.svd(J, compute_uv=False)
        sigma_min = float(np.min(s_vals)) if s_vals.size > 0 else 0.0
        lambda_reg = 1e-3 if sigma_min > 1e-1 else 1e-1

        dq_var = cp.Variable(self.nq)
        obj_primary = cp.sum_squares(W_sqrt @ (J @ dq_var - v_task)) + lambda_reg * cp.sum_squares(dq_var)
        prob1 = cp.Problem(cp.Minimize(obj_primary), [dq_var >= dq_min, dq_var <= dq_max])
        prob1.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if prob1.status in ["optimal", "optimal_inaccurate"] and dq_var.value is not None:
            dq1 = dq_var.value.reshape((self.nq, 1))
        else:
            # 失败回退：阻尼伪逆
            print("一级QP求解失败 回退阻尼伪逆法求解")
            J_pinv = self._compute_damped_pseudoinverse(J, min_sigma=1e-3)
            dq_fallback = J_pinv @ v_task
            dq_fallback = np.clip(dq_fallback, dq_min, dq_max)
            dq1 = dq_fallback.reshape((self.nq, 1))
        
        # 检查是否启用二级QP（完全按照原始代码）
        if not self.hqp_enabled and err_norm < self.switch_err_threshold:
            self.hqp_enabled = True
            null_dim = self.nq - rank
            if null_dim > 0:
                if self.perturbation_mode == 'fourier':
                    self.null_perturb = FourierNullspacePerturbation(
                        null_dim=null_dim, dt=self.dt, amp_scale=8.0
                    )
                else:
                    self.null_perturb = AdaptiveNullspacePerturbation(
                        null_dim=null_dim, dt=self.dt, amp_scale=8.0
                    )
                print(">>> HQP 二级扰动已启用")
        
        # 二级QP：零空间优化（完全按照原始代码）
        if self.hqp_enabled:
            # 使用之前计算的rank和null_basis，避免重复计算
            null_dim = self.nq - rank

            if null_dim > 0 and null_basis.size > 0:
                z = cp.Variable(null_dim)
                dq_expr = dq1.flatten() + null_basis @ z

                constraints2 = [dq_expr >= dq_min, dq_expr <= dq_max,
                                z >= -2.0, z <= 2.0]

                normalized = 2.0 * ((self.q.flatten() + dq_expr * self.dt) - self.q_mid) / (
                        (self.q_max - self.q_min) + 1e-8)
                obj_limits = cp.sum_squares(normalized)

                z_ref = self.null_perturb.step()
                obj_perturb = cp.sum_squares(z - z_ref)

                prob2 = cp.Problem(cp.Minimize(self.alpha_limit * obj_limits + self.beta_perturb * obj_perturb),
                                   constraints2)
                prob2.solve(solver=cp.OSQP, warm_start=True, verbose=False)
                if prob2.status in ["optimal", "optimal_inaccurate"]:
                    dq = dq1 + (null_basis @ z.value).reshape((self.nq, 1))
                else:
                    dq = dq1
            else:
                dq = dq1
        else:
            dq = dq1
        
        # 速度限幅（完全按照原始代码）
        max_step = 1.0  # 加快速度
        norm_dq = np.linalg.norm(dq)
        if norm_dq > max_step:
            dq = dq / norm_dq * max_step
        
        # 更新关节状态
        self.q = pin.integrate(self.robot.model, self.q, dq.flatten() * self.dt)
        self.q = pin.normalize(self.robot.model, self.q)
        self.dq = dq.flatten()
        
        # 更新性能指标
        solve_time = time.time() - start_time
        self._update_performance_metrics(err_norm, solve_time)
        
        # 返回结果
        result_info = {
            'error_norm': err_norm,
            'hqp_enabled': self.hqp_enabled,
            'dq_norm': norm_dq,
            'solve_time': solve_time,
            'rank': rank,
            'null_dim': self.nq - rank
        }
        
        return self.q, result_info
    
    def run_control_loop(self, goal_pose: pin.SE3, runtime: float = 20.0):
        """运行控制循环"""
        t0 = time.time()
        step = 0
        
        print(f"开始控制循环，目标运行时间: {runtime}秒")
        
        while time.time() - t0 < runtime:
            q, info = self.step(goal_pose)
            
            # 可视化
            if hasattr(self, 'viewer'):
                self.viewer.display(q)
            
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


class FourierNullspacePerturbation:
    """基于傅里叶级数的零空间扰动，更自然的周期性运动"""
    def __init__(self, null_dim: int, dt: float = 0.02, base_period: float = 6.0, num_harmonics: int = 1, amp_scale: float = 5.0):
        self.null_dim = null_dim
        self.dt = dt
        self.t = 0.0
        self.base_period = base_period
        self.num_harmonics = num_harmonics
        self.amp_scale = amp_scale
        # 为每个维度、每个谐波生成随机相位
        self.phases = np.random.uniform(0, 2*np.pi, size=(null_dim, num_harmonics))
        # 每个谐波的幅度衰减（高频更小）
        self.harmonic_scales = 1.0 / (np.arange(1, num_harmonics+1))

    def step(self) -> np.ndarray:
        self.t += self.dt
        omega = 2 * np.pi / self.base_period
        values = np.zeros(self.null_dim)
        for i in range(self.null_dim):
            s = 0.0
            for k in range(self.num_harmonics):
                freq = (k + 1)
                s += self.harmonic_scales[k] * np.sin(freq * omega * self.t + self.phases[i, k])
            values[i] = s
        return values * self.amp_scale * 0.5


def main():
    """主函数"""
    # 机器人模型路径
    urdf_path = osp.join(
        osp.dirname(__file__),
        "7dof_robstride",
        "robstride_right.urdf",
    )
    
    # 创建控制器
    arm = ImprovedHQPArm(urdf_path, target_frame_name="r_joint7", dt=0.02)
    
    # 设置目标位姿（尝试不同的方向）
    desired_rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    # 尝试不同的目标位置，看看哪个方向是正确的
    goal_pose = pin.SE3(desired_rot, np.array([0.5, 0.0, 0.0]))
    print(f"目标位姿设置: 旋转矩阵=\n{desired_rot}")
    print(f"目标位置: {goal_pose.translation}")
    print(f"目标旋转: {goal_pose.rotation}")
    
    # 可视化目标
    if hasattr(arm, 'viewer'):
        meshcat_shapes.frame(arm.viewer.viewer["target"], opacity=0.5)
        arm.viewer.viewer["target"].set_transform(goal_pose.np)
    
    # 运行控制循环
    arm.run_control_loop(goal_pose, runtime=200.0)


if __name__ == "__main__":
    main()

