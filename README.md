# 改进的HQP IK求解器：代码解析与公式化笔记

> 目标：把给定脚本的结构、数据流与核心数学写清楚，便于二次开发与调参。重点在 `ImprovedHQPArm.step()` 的控制逻辑，并将其抽象成可复用的公式与流程。

---

## 1. 工程概览（What & Why）

**功能**

- 通过 **HQP（分层二次规划）** 实现机械臂 IK：优先完成末端位姿任务，其次在零空间做“平滑扰动/限位居中”等优化。
- 结合 **Pinocchio** 做运动学、**OSQP** 做QP求解、**Meshcat** 可视化、**ROS2** 发布关节命令。
- 具备鲁棒性细节：SVD 估计秩与零空间、阻尼伪逆回退、自适应任务加权、速度/位置限幅、二级QP仅在误差足够小时启用。

**运行主线**

1. 载入 URDF → 生成 `RobotWrapper` 与目标帧 `FRAME_ID`。
2. 启动 ROS2 节点 `ROS2JointStatePublisher`，50 Hz 发布 `/right/ik_robstride_joint_cmd`。
3. `ImprovedHQPArm.run_control_loop()` 循环：每步调用 `step(goal_pose)` 计算 $dq$，积分得到新 $q$ 并可视化、发布。

---

## 2. 代码结构速览（Who does What）

- `ROS2 Node`：周期发布 `JointState`（仅位置）。
- `NullspacePerturbation` / `z_ref`：生成零空间扰动参考，使冗余自由度产生自然、平滑的周期运动（Fourier 或样条）。
- `QPWrapper`：通用 QP 包装（此处主体 QP 直接在 `step()` 用 cvxpy 组建）。
- `ImprovedHQPArm`：核心控制器，封装 Pinocchio、限位、求解与可视化；`step()` 实现 HQP 控制。

---

## 3. 关键数据与约定

- 关节向量：$q \in \mathbb{R}^{n}$，关节速度：$dq \in \mathbb{R}^{n}$。
- 末端相对误差采用 **李群/李代数**：$e \in \mathbb{R}^{6}$，由 $SE(3)$ 的对数映射得到（前3平移、后3旋转）。
- 任务雅可比：$J \in \mathbb{R}^{6 \times n}$，Pinocchio 计算帧雅可比。
- 限位：
  - 位置：$q\_{min}, q\_{max}$；中点：$q\_{mid} = 0.5 (q\_{min} + q\_{max})$。
  - 速度：$dq \in [dq\_{min}, dq\_{max}]$，其中 $dq\_{min/dq\_{max}}$ 由位置限位 + 最大速度共同裁剪。

---

## 4. 几何误差与任务速度（Step 1）

设当前末端位姿 ${}^oT_f$，目标位姿 ${}^oT_g$（代码中 `oMf` 与 `goal_pose`）。误差采用：

$$
T_e = {}^oT_g\,{}^oT_f^{-1}, \quad e = \log(T_e) \in \mathfrak{se}(3) \simeq \mathbb{R}^6.
$$

其中 $e = [e\_p; e\_\omega]$（平移与旋转误差）。采用**一阶**任务速度律：

$$
v_{task} = -K_p\, e, \quad (K_p > 0).
$$

> 代码：`err = pin.log(goal_pose.actInv(oMf)).vector`，`v_task = -self.Kp_task * err`。

---

## 5. 自适应任务加权（Step 2）

根据位置/姿态误差大小调节权重：

$$
W = \operatorname{diag}(w_p, w_p, w_p, w_o, w_o, w_o),\\
\quad w_p = 1 + 2\, \frac{\|e_p\|}{\|e_p\|+\|e_\omega\|+\varepsilon},\quad
w_o = 1 + 2\, \frac{\|e_\omega\|}{\|e_p\|+\|e_\omega\|+\varepsilon}.
$$

优化中使用 $W^{1/2}$：`W_sqrt = sqrt(W)`。

---

## 6. 速度约束与阻尼（Step 3）

以时间步长 $\Delta t$ 将位置限位转为速度窗：

$$
dq_{min} = \max\!\left( \frac{q_{min}-q}{\Delta t},\; -dq_{max}\right),\quad
dq_{max} = \min\!\left( \frac{q_{max}-q}{\Delta t},\; dq_{max}\right).
$$

另对主问题加入 $\ell\_2$ 正则以改善病态：$\lambda \cdot |dq|^2$，其中 $\lambda$ 随 $\sigma\_{min}(J)$ 自适应（`\sigma_{min}>1e-2 → λ=1e-4` 否则 `λ=1e-2`）。

---

## 7. 一级QP：任务优先（Step 4）

**目标**：让 $J dq \approx v\_{task}$，兼顾加权与阻尼，并满足速度限位：
$$
\begin{aligned}
\min_{dq \in \mathbb{R}^n}&\; \left\| W^{1/2} (J\, dq - v_{task}) \right\|_2^2 + \lambda\,\|dq\|_2^2,\\
\text{s.t.}&\; dq_{min} \le dq \le dq_{max}.
\end{aligned}
$$

> 代码：`obj_primary = ||W_sqrt @ (J @ dq - v_task)||² + λ||dq||²`，约束 `dq∈[dq_min,dq_max]`，OSQP 求解。

**回退机制（阻尼伪逆）**：若 QP 求解失败/不可靠，则
$$
J = U\,\Sigma\,V^\top,\quad
J^{\#}_\text{damped} = V\,\operatorname{diag}\!\left(\frac{\sigma_i}{\sigma_i^2 + \lambda_d}\right) U^\top,\\
\Rightarrow\; dq = \operatorname{clip}\big(J^{\#}_\text{damped}\, v_{task},\; dq_{min}, dq_{max}\big),
$$

其中 `λ_d = (σ_{min_target} / σ_{min}(J))^2`，代码里 `σ_{min_target}=1e−3`。

---

## 8. HQP 启动条件（Step 5）

当任务误差足够小，才启用二级优化：

$$
\|e\| < \texttt{switch\_err\_threshold} = 10^{-3} \;\Rightarrow\; \text{enable Level-2}.
$$

并用 SVD 估计 $\operatorname{rank}(J)$ 与零空间维数 $n_0 = n - \operatorname{rank}(J)$，构造零空间基 $N \in \mathbb{R}^{n\times n_0}$。

---

## 9. 二级QP：零空间优化（Step 6）

在保持一级解 $dq_1$ 不变的前提下，用零空间变量 $z \in \mathbb{R}^{n_0}$ 做微调：

$$
\begin{aligned}
& dq = dq_1 + N z, \\
& dq_{min} \le dq \le dq_{max},\quad -2 \le z \le 2.
\end{aligned}
$$

**目标函数**由两部分组成：

1. **限位居中**（下一步位置相对中点的归一化偏差）：

$$
q^{+} = q + dq\,\Delta t,\quad
\tilde{q} = \frac{2\,(q^{+} - q_{mid})}{(q_{max} - q_{min}) + \varepsilon},\quad
J_\text{limits} = \|\tilde{q}\|_2^2.
$$

2. **扰动跟踪**（鼓励自然“呼吸”运动）：

$$
J_\text{perturb} = \| z - z_{ref}\|_2^2,\quad z_{ref} = \text{Fourier/Spline 产生的参考}.
$$

合成目标：

$$
\min_{z}\; \alpha\, J_\text{limits} + \beta\, J_\text{perturb},\quad \alpha=\texttt{alpha\_limit},\; \beta=0.3.
$$

> 代码：`dq_expr = dq1 + N @ z`，`normalized = …`，`obj = α·sum_squares(normalized) + 0.3·||z - z_ref||²`。

---

## 10. 步长限幅与积分（Step 7）

为避免过大速度步长：

$$
\|dq\|_2 > dq_{\text{maxstep}} \; \Rightarrow\; dq \leftarrow dq\, \frac{dq_{\text{maxstep}}}{\|dq\|_2},\quad dq_{\text{maxstep}} = 1.0.
$$

随后用 Pinocchio 的 **指数积分** 与 **归一化** 更新：

$$
q \leftarrow \operatorname{integrate}(q, dq\,\Delta t),\quad q \leftarrow \operatorname{normalize}(q).
$$

并将 $q$ 发布到 ROS2 话题。

---

## 11. `step()` 全流程伪代码

```text
Input: goal_pose, state (q), params (Kp, dt, limits, …)
1) FK & frame placement
2) e = log(goal * inv(current));  v_task = -Kp * e
3) J, N, rank ← SVD(J(q));  build dq bounds from q_min/max and dq_max
4) W ← adaptive weights from |e_p|, |e_ω|; λ ← f(σ_min(J))
5) Solve Level-1 QP:  min ||W^{1/2}(J dq - v_task)||^2 + λ||dq||^2,  s.t. dq∈[dq_min,dq_max]
   if fail → dq ← damped-pinv(J) * v_task, clip to bounds
6) if ||e|| < threshold: enable HQP
   if enabled and dim(N)>0:
      z_ref ← Fourier/Spline
      dq = dq1 + N z
      Solve Level-2 QP:  min α·limit-centering(q + dq·dt) + β·||z - z_ref||^2
                         s.t. dq bounds & z bounds
7) norm-limit dq;  q ← integrate(q, dq·dt); normalize(q)
8) publish(q);  record metrics;  return q and info
```

---

## 12. 组件细化说明

### 12.1 雅可比与零空间（鲁棒）

- 用 `SVD(J)` 估计秩：`rank = #{σ_i > 1e-8}`。
- 零空间基 $N$ 由 $V^\top$ 的后若干列构成：`N = V[:, rank:]`（代码等价实现）。
- 失败兜底：若计算异常，返回 $J = I\_{6\times n}$。

### 12.2 自适应扰动 $z\_{ref}$

- **Fourier**：多个谐波衰减叠加，随机初相，平滑自然。
- **Spline**：周期样条的控制点随机生成，可根据误差/限位违反适当缩放幅度。
- 二级 QP 只在主任务完成后启用，避免干扰收敛。

### 12.3 自适应权重与正则

- $W$ 使得位置/姿态误差占比大的一侧获得更高权重。
- $\lambda$ 随 $\sigma_{min}(J)$ 自适应：越接近奇异，越大阻尼。

---

## 13. ROS2 与可视化

- 话题：`/right/ik_robstride_joint_cmd`（`JointState`，仅 `position`）。
- 发布频率：50 Hz；每 100 次打印一次当前 joint。
- 可视化：`Meshcat` 显示当前姿态，并绘制 `target` 坐标系（半透明）。

---

## 14. 调参与实战建议

- **Kp\_task（默认 4.0）**：过小收敛慢，过大易振荡；配合 $\lambda$ 使用。
- **switch\_err\_threshold（1e-3）**：越小，越晚进入零空间优化；任务优先更强。
- **alpha\_limit（10.0）**：越大越强制靠近中位，适合防限位；过大可能限制可达运动。
- **扰动幅度 amp\_scale**：冗余维度较多时可稍大；靠近限位或误差回升时应减小。
- **速度上限 dq\_max**：来自模型或手动设定，过小会影响收敛速度；过大可能冲击。
- **时间步长 dt（0.02s）**：越小越稳定但算力需求增大。

---

## 15. 常见问题排查

- **QP 不可行**：检查 $dq\_{min} > dq\_{max}$ 的关节；或 $W$ 过大、$v\_{task}$ 过激。
- **卡奇异**：观察 $\sigma\_{min}(J)$；提高 $\lambda$ 或稍改目标位姿，允许零空间调整；也可增大步长限幅保护。
- **进入 HQP 后误差回升**：适当减小扰动幅度或增大 $\alpha\_{limit}$；必要时提高启用阈值。
- **ROS2 无接收**：确认话题名、joint name 列表与下游控制器匹配；检查时间戳与 QoS。

---

## 16. 扩展与改造点

- **多任务 HQP**：将姿态与位置拆为不同层级（例如先位置后姿态），或加入末端速度/加速度跟踪层。
- **碰撞/障碍物**：在二级目标加入距离场势能或不等式约束（需要额外几何库）。
- **动力学一致性**：把 `dq` 扩展到 `τ` 层（考虑质量矩阵 `M(q)` 的加权），形成 WBHQP。
- **实时性**：预热 `OSQP`，固定稀疏结构，减少 GC/内存分配；将部分计算（SVD/雅可比）向量化或并行。

