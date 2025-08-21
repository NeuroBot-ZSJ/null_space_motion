#!/usr/bin/env python3
"""
HQP性能分析工具
分析求解器性能，提供优化建议
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import json
import os
import os.path as osp
from matplotlib import rcParams, font_manager

# 自动查找系统里可用的中文字体
zh_font = None
for f in font_manager.findSystemFonts(fontpaths=None, fontext='ttf'):
    if any(name in f for name in ["SimHei", "SimSun", "Microsoft YaHei", "PingFang", "Heiti", "NotoSansCJK"]):
        zh_font = f
        break

if zh_font:
    rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=zh_font).get_name()]
    rcParams['axes.unicode_minus'] = False
    print(f"已加载中文字体: {rcParams['font.sans-serif'][0]}")
else:
    print("⚠️ 没找到中文字体，请安装 SimHei / Noto Sans CJK 之类的字体")
# 允许直接接入改进的HQP求解器以进行真实性能测试

try:
    from improved_hqp_ik_solver import ImprovedHQPArm
    import pinocchio as pin
except Exception:
    ImprovedHQPArm = None
    pin = None


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    # 时间指标
    solve_times: List[float]
    total_time: float
    avg_solve_time: float
    max_solve_time: float
    min_solve_time: float
    
    # 任务性能
    task_errors: List[float]
    final_error: float
    convergence_time: float
    steady_state_error: float
    
    # 求解器性能
    success_rate: float
    infeasible_count: int
    unbounded_count: int
    solver_error_count: int
    
    # 零空间性能
    nullspace_utilization: float
    perturbation_effectiveness: float
    
    # 关节性能
    joint_limit_violations: List[float]
    max_joint_velocity: float
    avg_joint_velocity: float


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.metrics = None
        self.analysis_results = {}
        # 运行时缓存：用于插值前/后的对比
        self.last_runtime_data = None
        self.last_interpolation = None
    
    def analyze_performance(self, performance_data: Dict[str, Any]) -> PerformanceMetrics:
        """分析性能数据"""
        # 提取数据
        solve_times = performance_data.get('solve_times', [])
        task_errors = performance_data.get('task_errors', [])
        joint_velocities = performance_data.get('joint_velocities', [])
        solver_statuses = performance_data.get('solver_statuses', [])
        nullspace_usage = performance_data.get('nullspace_usage', [])
        
        # 计算时间指标
        total_time = sum(solve_times) if solve_times else 0
        avg_solve_time = np.mean(solve_times) if solve_times else 0
        max_solve_time = max(solve_times) if solve_times else 0
        min_solve_time = min(solve_times) if solve_times else 0
        
        # 计算任务性能
        final_error = task_errors[-1] if task_errors else float('inf')
        convergence_time = self._compute_convergence_time(task_errors)
        steady_state_error = np.mean(task_errors[-50:]) if len(task_errors) >= 50 else final_error
        
        # 计算求解器性能
        total_solves = len(solver_statuses)
        success_count = sum(1 for status in solver_statuses if status == 'success')
        infeasible_count = sum(1 for status in solver_statuses if status == 'infeasible')
        unbounded_count = sum(1 for status in solver_statuses if status == 'unbounded')
        solver_error_count = sum(1 for status in solver_statuses if status == 'solver_error')
        
        success_rate = success_count / total_solves if total_solves > 0 else 0
        
        # 计算零空间性能
        nullspace_utilization = np.mean(nullspace_usage) if nullspace_usage else 0
        perturbation_effectiveness = self._compute_perturbation_effectiveness(
            task_errors, nullspace_usage
        )
        
        # 计算关节性能
        joint_limit_violations = performance_data.get('joint_limit_violations', [])
        max_joint_velocity = max(joint_velocities) if joint_velocities else 0
        avg_joint_velocity = np.mean(joint_velocities) if joint_velocities else 0
        
        # 创建性能指标对象
        self.metrics = PerformanceMetrics(
            solve_times=solve_times,
            total_time=total_time,
            avg_solve_time=avg_solve_time,
            max_solve_time=max_solve_time,
            min_solve_time=min_solve_time,
            task_errors=task_errors,
            final_error=final_error,
            convergence_time=convergence_time,
            steady_state_error=steady_state_error,
            success_rate=success_rate,
            infeasible_count=infeasible_count,
            unbounded_count=unbounded_count,
            solver_error_count=solver_error_count,
            nullspace_utilization=nullspace_utilization,
            perturbation_effectiveness=perturbation_effectiveness,
            joint_limit_violations=joint_limit_violations,
            max_joint_velocity=max_joint_velocity,
            avg_joint_velocity=avg_joint_velocity
        )
        
        return self.metrics
    
    def _as_np(self, x):
        return np.array(x) if not isinstance(x, np.ndarray) else x
    
    def _compute_smoothness_metrics(self, t: np.ndarray, q: np.ndarray, dq: np.ndarray) -> Dict[str, Any]:
        """计算平滑度与动态性能指标：速度/加速度/加加速度（jerk）"""
        if t.size == 0 or q.size == 0:
            return {
                'rms_velocity': 0.0,
                'peak_velocity': 0.0,
                'rms_acceleration': 0.0,
                'peak_acceleration': 0.0,
                'rms_jerk': 0.0,
                'peak_jerk': 0.0,
                'mean_position_step': 0.0
            }
        t = self._as_np(t).flatten()
        q = self._as_np(q)
        dq = self._as_np(dq)
        # 对齐长度
        n = min(q.shape[0], dq.shape[0], t.shape[0])
        q = q[:n]
        dq = dq[:n]
        t = t[:n]
        # 加速度/加加速度
        ddq = np.gradient(dq, t, axis=0)
        dddq = np.gradient(ddq, t, axis=0)
        # 统计指标（聚合所有关节）
        rms_velocity = float(np.sqrt(np.mean(dq**2)))
        peak_velocity = float(np.max(np.abs(dq)))
        rms_acceleration = float(np.sqrt(np.mean(ddq**2)))
        peak_acceleration = float(np.max(np.abs(ddq)))
        rms_jerk = float(np.sqrt(np.mean(dddq**2)))
        peak_jerk = float(np.max(np.abs(dddq)))
        # 平均相邻位置步进（越小越平滑）
        dq_pos = np.diff(q, axis=0)
        mean_position_step = float(np.mean(np.linalg.norm(dq_pos, axis=1))) if dq_pos.shape[0] > 0 else 0.0
        return {
            'rms_velocity': rms_velocity,
            'peak_velocity': peak_velocity,
            'rms_acceleration': rms_acceleration,
            'peak_acceleration': peak_acceleration,
            'rms_jerk': rms_jerk,
            'peak_jerk': peak_jerk,
            'mean_position_step': mean_position_step
        }
    
    def compare_pre_post_interpolation(self, pre_data: Dict[str, Any] = None,
                                       post_interp: Dict[str, Any] = None) -> Dict[str, Any]:
        """对比插值前(原dt轨迹)与插值后(1kHz)的平滑度与动态性能"""
        if pre_data is None:
            pre_data = self.last_runtime_data
        if post_interp is None:
            post_interp = self.last_interpolation
        if pre_data is None or post_interp is None:
            raise RuntimeError("缺少对比数据，请先运行 analyze_solver_runtime 或提供数据参数")
        # 原始
        t_pre = np.array(pre_data.get('timestamps', []))
        q_pre = np.array(pre_data.get('q_samples', []))
        dq_pre = np.array(pre_data.get('dq_samples', []))
        m_pre = self._compute_smoothness_metrics(t_pre, q_pre, dq_pre)
        # 插值
        t_post = np.array(post_interp.get('t_1khz', []))
        q_post = np.array(post_interp.get('q_1khz', []))
        dq_post = np.array(post_interp.get('dq_1khz', []))
        m_post = self._compute_smoothness_metrics(t_post, q_post, dq_post)
        # 改善度(正值表示插值后更平滑/更小)
        def improve(a, b):
            return float((a - b) / a) if a > 1e-12 else 0.0
        comparison = {
            'pre': m_pre,
            'post': m_post,
            'improvements': {
                'rms_acceleration_reduction': improve(m_pre['rms_acceleration'], m_post['rms_acceleration']),
                'peak_acceleration_reduction': improve(m_pre['peak_acceleration'], m_post['peak_acceleration']),
                'rms_jerk_reduction': improve(m_pre['rms_jerk'], m_post['rms_jerk']),
                'peak_jerk_reduction': improve(m_pre['peak_jerk'], m_post['peak_jerk']),
                'mean_position_step_reduction': improve(m_pre['mean_position_step'], m_post['mean_position_step'])
            }
        }
        self.analysis_results['interpolation_comparison'] = comparison
        return comparison

    def analyze_solver_runtime(self, runtime: float = 10.0, dt: float = 0.02,
                               visualize: bool = False,
                               urdf_relative: str = osp.join("7dof_robstride", "robstride_right.urdf"),
                               target_frame: str = "r_joint7") -> PerformanceMetrics:
        """直接运行改进的求解器，收集真实数据并分析"""
        if ImprovedHQPArm is None or pin is None:
            raise RuntimeError("找不到ImprovedHQPArm或pinocchio，请确保依赖已安装且文件可导入")
        urdf_path = osp.join(osp.dirname(__file__), urdf_relative)
        arm = ImprovedHQPArm(urdf_path, target_frame_name=target_frame, dt=dt, visualize=visualize)
        desired_rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        goal_pose = pin.SE3(desired_rot, np.array([0.5, 0.0, 0.0]))
        arm.run_control_loop(goal_pose, runtime=runtime)
        perf = arm.get_performance_data()
        metrics = self.analyze_performance(perf)
        # 缓存原始运行数据
        self.last_runtime_data = perf
        # 如果有轨迹日志，自动生成1kHz插值结果并保存
        try:
            # 优先使用在线插值缓存；若为空，再退回离线全局插值
            online = arm.get_online_interpolated_data()
            use_online = online['t_1khz'].size > 0
            interp = online if use_online else arm.interpolate_trajectory_1khz()
            if (interp.get('t_1khz', []) is not None) and (len(interp['t_1khz']) > 0):
                np.savez("trajectory_1khz.npz", t=np.array(interp['t_1khz']),
                         q=np.array(interp['q_1khz']), dq=np.array(interp['dq_1khz']))
                print("1kHz轨迹已保存为 trajectory_1khz.npz")
            self.last_interpolation = interp
        except Exception as e:
            print(f"1kHz轨迹插值失败: {e}")
        return metrics
    
    def _compute_convergence_time(self, task_errors: List[float], 
                                 threshold: float = 1e-3) -> float:
        """计算收敛时间"""
        if not task_errors:
            return float('inf')
        
        for i, error in enumerate(task_errors):
            if error < threshold:
                return i * 0.02  # 假设时间步长为0.02s
        
        return float('inf')
    
    def _compute_perturbation_effectiveness(self, task_errors: List[float], 
                                          nullspace_usage: List[bool]) -> float:
        """计算扰动有效性"""
        if len(task_errors) < 2 or len(nullspace_usage) < 2:
            return 0.0
        
        # 计算启用零空间扰动前后的误差变化
        errors_with_nullspace = []
        errors_without_nullspace = []
        
        for i, use_nullspace in enumerate(nullspace_usage):
            if i < len(task_errors):
                if use_nullspace:
                    errors_with_nullspace.append(task_errors[i])
                else:
                    errors_without_nullspace.append(task_errors[i])
        
        if not errors_with_nullspace or not errors_without_nullspace:
            return 0.0
        
        avg_error_with = np.mean(errors_with_nullspace)
        avg_error_without = np.mean(errors_without_nullspace)
        
        if avg_error_without == 0:
            return 0.0
        
        # 扰动有效性 = (无扰动误差 - 有扰动误差) / 无扰动误差
        effectiveness = (avg_error_without - avg_error_with) / avg_error_without
        return max(0.0, min(1.0, effectiveness))  # 限制在[0,1]范围内
    
    def generate_optimization_recommendations(self) -> List[str]:
        """生成优化建议"""
        if not self.metrics:
            return ["请先运行性能分析"]
        
        recommendations = []
        
        # 求解时间优化建议
        if self.metrics.avg_solve_time > 0.01:  # 10ms
            recommendations.append("求解时间过长，建议：")
            recommendations.append("  - 降低求解精度 (eps_abs, eps_rel)")
            recommendations.append("  - 减少最大迭代次数")
            recommendations.append("  - 使用更快的求解器 (如ECOS)")
        
        # 任务性能优化建议
        if self.metrics.final_error > 1e-3:
            recommendations.append("任务误差过大，建议：")
            recommendations.append("  - 增加任务增益 Kp_task")
            recommendations.append("  - 检查目标位姿是否可达")
            recommendations.append("  - 调整关节限位约束")
        
        # 求解器稳定性建议
        if self.metrics.success_rate < 0.95:
            recommendations.append("求解器成功率低，建议：")
            recommendations.append("  - 增加正则化项")
            recommendations.append("  - 调整约束边界")
            recommendations.append("  - 检查问题数值条件")
        
        # 零空间优化建议
        if self.metrics.nullspace_utilization < 0.5:
            recommendations.append("零空间利用率低，建议：")
            recommendations.append("  - 调整扰动幅度和频率")
            recommendations.append("  - 优化扰动权重 alpha_perturb")
        
        # 关节性能建议
        if self.metrics.max_joint_velocity > 3.0:
            recommendations.append("关节速度过高，建议：")
            recommendations.append("  - 降低速度限位 dq_max")
            recommendations.append("  - 增加速度平滑")
        
        return recommendations
    
    def plot_performance_analysis(self, save_path: str = None):
        """绘制性能分析图表"""
        if not self.metrics:
            print("请先运行性能分析")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('HQP性能分析报告', fontsize=16)
        
        # 1. 求解时间
        axes[0, 0].plot(self.metrics.solve_times)
        axes[0, 0].set_title('求解时间')
        axes[0, 0].set_xlabel('步数')
        axes[0, 0].set_ylabel('时间 (s)')
        axes[0, 0].grid(True)
        
        # 2. 任务误差
        axes[0, 1].plot(self.metrics.task_errors)
        axes[0, 1].set_title('任务误差')
        axes[0, 1].set_xlabel('步数')
        axes[0, 1].set_ylabel('误差')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # 3. 求解器状态分布
        status_counts = [
            self.metrics.success_rate * 100,
            self.metrics.infeasible_count,
            self.metrics.unbounded_count,
            self.metrics.solver_error_count
        ]
        status_labels = ['成功', '不可行', '无界', '求解器错误']
        axes[0, 2].pie(status_counts, labels=status_labels, autopct='%1.1f%%')
        axes[0, 2].set_title('求解器状态分布')
        
        # 4. 性能指标总结
        performance_text = f"""
总时间: {self.metrics.total_time:.3f}s
平均求解时间: {self.metrics.avg_solve_time*1000:.1f}ms
最终误差: {self.metrics.final_error:.2e}
收敛时间: {self.metrics.convergence_time:.2f}s
零空间利用率: {self.metrics.nullspace_utilization:.1%}
扰动有效性: {self.metrics.perturbation_effectiveness:.1%}
        """
        axes[1, 0].text(0.1, 0.5, performance_text, transform=axes[1, 0].transAxes,
                        fontsize=10, verticalalignment='center')
        axes[1, 0].set_title('性能指标总结')
        axes[1, 0].axis('off')
        
        # 5. 关节限位违反
        if self.metrics.joint_limit_violations:
            axes[1, 1].plot(self.metrics.joint_limit_violations)
            axes[1, 1].set_title('关节限位违反')
            axes[1, 1].set_xlabel('步数')
            axes[1, 1].set_ylabel('违反程度')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, '无关节限位违反', 
                           transform=axes[1, 1].transAxes,
                           ha='center', va='center')
            axes[1, 1].set_title('关节限位违反')
            axes[1, 1].axis('off')
        
        # 6. 优化建议
        recommendations = self.generate_optimization_recommendations()
        rec_text = '\n'.join(recommendations[:8])  # 限制显示行数
        axes[1, 2].text(0.1, 0.5, rec_text, transform=axes[1, 2].transAxes,
                        fontsize=9, verticalalignment='center')
        axes[1, 2].set_title('优化建议')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"性能分析图表已保存到: {save_path}")
        
        plt.show()
    
    def plot_interpolation_comparison(self, save_path: str = None, joint_index: int = 0):
        """绘制插值前后轨迹的对比图与平滑度柱状图"""
        if self.last_runtime_data is None or self.last_interpolation is None:
            print("缺少数据，请先运行 analyze_solver_runtime 或提供数据")
            return
        # 时间与轨迹
        t_pre = np.array(self.last_runtime_data.get('timestamps', []))
        q_pre = np.array(self.last_runtime_data.get('q_samples', []))
        dq_pre = np.array(self.last_runtime_data.get('dq_samples', []))
        t_post = np.array(self.last_interpolation.get('t_1khz', []))
        q_post = np.array(self.last_interpolation.get('q_1khz', []))
        dq_post = np.array(self.last_interpolation.get('dq_1khz', []))
        comp = self.analysis_results.get('interpolation_comparison') or self.compare_pre_post_interpolation()
        # 准备图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle('插值前后轨迹对比分析', fontsize=16)
        # 1. 位置轨迹叠加
        if q_pre.size > 0 and q_post.size > 0:
            axes[0, 0].plot(t_pre, q_pre[:, joint_index], 'o-', label='原始(50Hz)', markersize=3)
            axes[0, 0].plot(t_post, q_post[:, joint_index], '-', label='插值(1kHz)', linewidth=1)
            axes[0, 0].set_title(f'关节{joint_index} 位置轨迹')
            axes[0, 0].set_xlabel('时间 (s)')
            axes[0, 0].set_ylabel('关节角 (rad)')
            axes[0, 0].grid(True)
            axes[0, 0].legend()
        # 2. 速度轨迹叠加
        if dq_pre.size > 0 and dq_post.size > 0:
            axes[0, 1].plot(t_pre, dq_pre[:, joint_index], 'o-', label='原始(50Hz)', markersize=3)
            axes[0, 1].plot(t_post, dq_post[:, joint_index], '-', label='插值(1kHz)', linewidth=1)
            axes[0, 1].set_title(f'关节{joint_index} 速度轨迹')
            axes[0, 1].set_xlabel('时间 (s)')
            axes[0, 1].set_ylabel('关节角速度 (rad/s)')
            axes[0, 1].grid(True)
            axes[0, 1].legend()
        # 3. 平滑度(RMS/峰值)柱状图
        labels = ['acc_rms', 'acc_peak', 'jerk_rms', 'jerk_peak', 'pos_step']
        pre_vals = [comp['pre']['rms_acceleration'], comp['pre']['peak_acceleration'],
                    comp['pre']['rms_jerk'], comp['pre']['peak_jerk'], comp['pre']['mean_position_step']]
        post_vals = [comp['post']['rms_acceleration'], comp['post']['peak_acceleration'],
                     comp['post']['rms_jerk'], comp['post']['peak_jerk'], comp['post']['mean_position_step']]
        x = np.arange(len(labels))
        width = 0.35
        axes[1, 0].bar(x - width/2, pre_vals, width, label='原始')
        axes[1, 0].bar(x + width/2, post_vals, width, label='插值')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(labels)
        axes[1, 0].set_title('平滑度与动态指标对比')
        axes[1, 0].grid(True, axis='y')
        axes[1, 0].legend()
        # 4. 改善率文本
        impr = comp['improvements']
        text = (
            f"RMS加速度降低: {impr['rms_acceleration_reduction']:.1%}\n"
            f"峰值加速度降低: {impr['peak_acceleration_reduction']:.1%}\n"
            f"RMS加加速度降低: {impr['rms_jerk_reduction']:.1%}\n"
            f"峰值加加速度降低: {impr['peak_jerk_reduction']:.1%}\n"
            f"平均位置步进降低: {impr['mean_position_step_reduction']:.1%}"
        )
        axes[1, 1].text(0.1, 0.5, text, transform=axes[1, 1].transAxes,
                        fontsize=11, va='center')
        axes[1, 1].axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"插值对比图已保存到: {save_path}")
        plt.show()
    
    def save_analysis_report(self, filename: str = "performance_report.json"):
        """保存分析报告"""
        if not self.metrics:
            print("请先运行性能分析")
            return
        
        report = {
            'summary': {
                'total_time': self.metrics.total_time,
                'avg_solve_time': self.metrics.avg_solve_time,
                'final_error': self.metrics.final_error,
                'success_rate': self.metrics.success_rate,
                'nullspace_utilization': self.metrics.nullspace_utilization
            },
            'detailed_metrics': {
                'solve_times': self.metrics.solve_times,
                'task_errors': self.metrics.task_errors,
                'joint_limit_violations': self.metrics.joint_limit_violations
            },
            'recommendations': self.generate_optimization_recommendations(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"性能分析报告已保存到: {filename}")
    
    def compare_configurations(self, configs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """比较不同配置的性能"""
        comparison = {}
        
        for i, config_data in enumerate(configs_data):
            config_name = config_data.get('name', f'配置{i+1}')
            metrics = self.analyze_performance(config_data['performance'])
            
            comparison[config_name] = {
                'avg_solve_time': metrics.avg_solve_time,
                'final_error': metrics.final_error,
                'success_rate': metrics.success_rate,
                'nullspace_utilization': metrics.nullspace_utilization,
                'convergence_time': metrics.convergence_time
            }
        
        return comparison


def create_sample_performance_data() -> Dict[str, Any]:
    """创建示例性能数据用于测试"""
    np.random.seed(42)
    
    n_steps = 1000
    
    # 模拟求解时间（逐渐改善）
    base_time = 0.015
    solve_times = [base_time + 0.005 * np.exp(-i/200) + 0.002 * np.random.randn() 
                   for i in range(n_steps)]
    solve_times = [max(0.001, t) for t in solve_times]
    
    # 模拟任务误差（指数收敛）
    task_errors = [0.1 * np.exp(-i/100) + 0.001 * np.random.randn() 
                   for i in range(n_steps)]
    task_errors = [max(1e-6, abs(e)) for e in task_errors]
    
    # 模拟求解器状态
    success_rate = 0.98
    solver_statuses = ['success' if np.random.random() < success_rate else 'infeasible' 
                       for _ in range(n_steps)]
    
    # 模拟零空间使用
    nullspace_usage = [i > 100 for i in range(n_steps)]  # 100步后启用
    
    # 模拟关节速度
    joint_velocities = [2.0 + 0.5 * np.random.randn() for _ in range(n_steps)]
    joint_velocities = [max(0.1, abs(v)) for v in joint_velocities]
    
    # 模拟关节限位违反
    joint_limit_violations = [0.01 * np.random.random() for _ in range(n_steps)]
    
    return {
        'solve_times': solve_times,
        'task_errors': task_errors,
        'solver_statuses': solver_statuses,
        'nullspace_usage': nullspace_usage,
        'joint_velocities': joint_velocities,
        'joint_limit_violations': joint_limit_violations
    }


def main():
    """主函数 - 演示性能分析工具"""
    print("=== HQP性能分析工具：改进求解器实测 & 插值对比 ===")
    analyzer = PerformanceAnalyzer()
    try:
        metrics = analyzer.analyze_solver_runtime(runtime=20.0, dt=0.02, visualize=False)
    except Exception as e:
        print(f"直接运行改进求解器失败，回退到示例数据: {e}")
        metrics = analyzer.analyze_performance(create_sample_performance_data())
    print(f"\n关键性能指标:")
    print(f"平均求解时间: {metrics.avg_solve_time*1000:.1f}ms")
    print(f"最终任务误差: {metrics.final_error:.2e}")
    print(f"求解成功率: {metrics.success_rate:.1%}")
    print(f"零空间利用率: {metrics.nullspace_utilization:.1%}")
    print(f"收敛时间: {metrics.convergence_time:.2f}s")
    print(f"\n优化建议:")
    for rec in analyzer.generate_optimization_recommendations():
        print(f"  {rec}")
    print(f"\n正在生成性能分析图表...")
    analyzer.plot_performance_analysis("hqp_performance_analysis.png")
    # 插值前后对比
    try:
        comp = analyzer.compare_pre_post_interpolation()
        print("\n插值前后平滑度对比(聚合):")
        print(f"  RMS加速度: 原始={comp['pre']['rms_acceleration']:.3e}, 插值={comp['post']['rms_acceleration']:.3e}")
        print(f"  峰值加速度: 原始={comp['pre']['peak_acceleration']:.3e}, 插值={comp['post']['peak_acceleration']:.3e}")
        print(f"  RMS加加速度: 原始={comp['pre']['rms_jerk']:.3e}, 插值={comp['post']['rms_jerk']:.3e}")
        print(f"  峰值加加速度: 原始={comp['pre']['peak_jerk']:.3e}, 插值={comp['post']['peak_jerk']:.3e}")
        print(f"  平均位置步进: 原始={comp['pre']['mean_position_step']:.3e}, 插值={comp['post']['mean_position_step']:.3e}")
        analyzer.plot_interpolation_comparison("interp_comparison.png", joint_index=0)
    except Exception as e:
        print(f"插值前后对比失败: {e}")
    analyzer.save_analysis_report("hqp_performance_report.json")
    print(f"\n分析完成！")


if __name__ == "__main__":
    main()







