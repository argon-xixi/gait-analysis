# gait-analysis

该目录是本仓库中“步态分析”相关的自定义实现与主要改进点，核心入口为：

- `infer.py`：主推理脚本（从图片序列推理 → 关键点/分割 → 后处理 → 步态参数 → 曲线可视化 → 输出视频/np数据）
## 1.环境安装
mmpose部分参考[mmpose安装文档](https://mmpose.readthedocs.io/zh_CN/latest/installation.html)

sam部分参考[sam安装文档](segment-anything-main/README.md)



## 2. 推流流程
主要目录是[步态推理根目录](mmpose-main/yjh/infer.py)

### 1. 整体工作流（端到端）

以 `infer.py` 为主线，处理链路如下：

1. **输入准备**
   - 输入通常是一段行走视频。
   - 实际推理使用的是视频抽帧后的图片序列：`yjh/ori/<videoname>/pic/*.jpg`（文件名带帧号，例如 `00001.jpg`）。
   - 抽帧工具在 `pic2vid.py:getpic()`（脚本中目前默认注释掉，需要时手动打开调用）。

2. **人体检测 + 姿态估计（MMPose + MMDet）**
   - 通过 `demo/topdown_demo_with_mmdet_new.py` 中封装的 `mmpose` 类（此处当作一个 *det+pose* wrapper）完成：
     - MMDet: person detection → 得到 bbox
     - MMPose: top-down pose → 得到关键点
   - `infer.py` 取用 COCO 关键点中的髋/膝/踝：
     - left hip/knee/ankle: 11/13/15
     - right hip/knee/ankle: 12/14/16

3. **人体分割（Segment Anything, SAM）**
   - 使用检测到的 bbox 作为 prompt，调用 `SamPredictor.predict(box=...)` 得到人体 mask。
   - mask 会做一次 **闭运算**（`cv2.MORPH_CLOSE`）以填补空洞、连接断裂。
   - 输出：`yjh/ori/<videoname>/mask/*.jpg`

4. **几何特征提取（长度/宽度、足尖 toe 点）**
   - **大腿/小腿长度与宽度**：`length_and_width.py:length_and_width()`
     - 长度：两关键点（例如 hip-knee）欧式距离
     - 宽度：在该骨段中点处作垂线，与 mask 的骨架/边界求交，取近似左右边界点距离
   - **足尖 toe 点**：`footpoint_new.py:process_frame_footpoint()`
     - 默认用 `findfootpoint()`：在踝关节附近构造矩形 ROI，对 ROI 内轮廓做 PCA，取主方向投影最大的点作为 toe
     - 当左右脚 ROI **重叠** 时，toe 点容易串扰：
       - 使用“相对角速度模型”进行预测（保持 toe-knee 向量相对 ankle-knee 向量的角度连续）
       - 本项目对该部分做了额外的状态缓存 `previous_frame_data`

5. **角度与步态事件（HS/TO）**
   - `angle.py:find_angle()` 计算关节角：
     - ankle angle：以踝为顶点，(toe, knee) 两向量夹角
     - knee angle：以膝为顶点，(hip, ankle)
     - hip angle：以髋为顶点，使用一个“竖直参考点”构造与地面垂直的参考向量
   - HS/TO 推断：
     - 先对关键点 x 轨迹做 Savitzky-Golay 平滑
     - 用 `CubicSpline(...).derivative()` 求速度
     - 再用“局部极值”寻找 TO/HS（脚踝相对膝盖的前后摆动达到极小/极大时）

6. **可视化与视频输出**
   - `infer.py` 会生成 3x2 的关节角曲线图帧（`chart/*.jpg`）
   - `pred1/*.jpg`：在姿态可视化图上额外画 toe 点与 toe-ankle 连线
   - `pic2vid.py:getvid()`：把 `pred1/` 与 `chart/` 横向拼接，写出结果视频：`yjh/result/<videoname>.mp4`

7. **数据落盘**
   - `coords.npy`：每帧关键点坐标（shape: `[T, 8, 2]`，8个点为：左右 hip/knee/ankle + 左右 toe）
   - `results.npy`：每帧左右腿的角度/高度/长度宽度等（shape: `[T, 2, 10]`）

---

### 2. 目录结构速览

```text
yjh/
  infer.py                # 主推理脚本（核心入口）
  footpoint_new.py        # toe 点估计（含重叠时的预测策略）
  footpoint.py            # 早期版本（保留对比）
  length_and_width.py     # 大腿/小腿长度与宽度估计
  find_endpoints.py       # 骨架端点查找（用于宽度估计中的交点）
  angle.py                # 三点角度计算
  pic2vid.py              # 视频抽帧/拼接回视频
  markdown_new.py         # 结构化生成步态报告 Markdown（模板化）
  markdown.py             # 旧版 Markdown 生成示例
  video.py                # moviepy 裁剪/拼接实验脚本
  visual.py               # 早期实时曲线叠加实验（当前基本注释）

  ori/<videoname>/
    pic/                  # 输入帧
    mask/                 # SAM 分割 mask
    pred/                 # det+pose 可视化输出（来自 wrapper）
    pred1/                # 在 pred 上追加 toe 结果的可视化
    chart/                # 角度曲线图帧序列

  result/
    *_coords.npy
    *_results.npy
    <videoname>.mp4
```

---

### 3. 运行方式（建议）

#### 3.1 准备输入帧

将视频抽帧到：

```bash
python -c "from yjh.pic2vid import getpic; getpic('YOUR.mp4','mmpose-main/yjh/ori/<videoname>/pic/')"
```

保证帧名形如 `00001.jpg`，否则需要调整 `sort_by_number()` 的排序策略。

#### 3.2 配置路径

`infer.py` 内仍保留了作者的 `videoname`、`sam_checkpoint`、各输出目录等路径变量。

为了跨机器运行，建议你至少修改：

- `videoname = 'xxx'`
- `sam_checkpoint = '.../sam_vit_h_4b8939.pth'`
- `picpath/maskpath/predpath/...`（如果你不使用 Linux 的 `/home/yjh/...` 目录）

同时，如果你的 `segment-anything-main/` 不在默认可 import 路径下，可以通过环境变量追加：

```bash
# Linux/Mac
export SEGMENT_ANYTHING_DIR=/abs/path/to/segment-anything-main
export MMPOSE_DIR=/abs/path/to/mmpose-main
```

Windows PowerShell 示例：

```powershell
$env:SEGMENT_ANYTHING_DIR = "C:\\path\\to\\segment-anything-main"
$env:MMPOSE_DIR = "C:\\path\\to\\mmpose-main"
```

#### 3.3 执行推理

```bash
cd mmpose-main
python yjh/infer.py
```

---

### 4. 输出说明

执行完成后，通常会产出：

- `yjh/ori/<videoname>/mask/*.jpg`：人体 mask
- `yjh/ori/<videoname>/pred/*.jpg`：det+pose 可视化
- `yjh/ori/<videoname>/pred1/*.jpg`：追加 toe 点可视化
- `yjh/ori/<videoname>/chart/*.jpg`：角度曲线帧
- `yjh/result/<videoname>_coords.npy`：关键点时序
- `yjh/result/<videoname>_results.npy`：角度/长度宽度/高度时序
- `yjh/result/<videoname>.mp4`：拼接结果视频

---

### 5. 常见问题（FAQ）

1. **为什么要用 SAM？**
   - 姿态关键点只能给出骨架点，无法直接得到腿部“宽度”等形态学信息；
   - SAM mask 让我们可以在腿段处做横截面估计宽度，也能辅助 toe 点定位。

2. **左右脚重叠时 toe 点为什么会漂？**
   - 两个 ROI 相交时，PCA 的主方向可能由另一只脚的轮廓主导。
   - `footpoint_new.py` 用“相对角速度”与“平移近似”来保持 toe 轨迹连续。

3. **只想跑单张图片/单帧可以吗？**
   - 可以，但 `infer.py` 的下半部分（步态周期、速度、曲线生成）依赖时序数据。
   - 若只需要关键点+mask+toe，请裁剪主循环部分。

---


