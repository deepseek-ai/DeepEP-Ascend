# EPBuffer 兼容测试入口

本套测试用于保持 DeepEP-Ascend 官方接口及功能语义，并验证默认 MTE 优化的接入。它区分官方基线、候选关闭优化和候选自动优化；不切换源码分支，也不构建或安装库。

`test_ep_auto_compat.py` 的正确性矩阵与 `test_stream.py` 的流测试互补。前者逐操作同步，适合数值、metadata 和路径验证；后者验证异步流依赖与 Tensor 生命周期。开启 profiler 的运行不用于性能或 overlap 结论。

## 运行身份

| `--backend` | 实际要求 | 环境开关 |
| --- | --- | --- |
| `official` | 独立安装/加载官方 `3b25377`；不得把候选关闭优化称为官方 | 不修改诊断开关 |
| `candidate-disabled` | 安装/加载候选版本 | 设置 `EP_DISABLE_LOW_LATENCY_TRANSPORT=1` |
| `candidate-auto` | 安装/加载候选版本 | 移除上述开关 |

测试记录实际导入的 `deep_ep` 路径、扩展二进制 SHA256、可获取的源码 commit/工作区状态、Python/Torch/torch_npu、相关运行文件及测试工具哈希。`--build-label` 只是人工标签。`--expected-source-commit` 从实际导入包所在 Git 仓核验；无可核验源码时会报错，不会把标签当作验证结果。

官方没有候选新增的 `low_latency_mode/is_low_latency` 属性；公共返回值、metadata、权重和事件检查保持一致。候选运行另外要求这两个标记存在且为 False，避免误入独立 LL。固定官方运行若发现候选标记会报错，提示加载错库。

在官方环境运行时，把下列四个文件作为一个外部测试目录使用即可，无需修改官方通信实现：

```text
test_ep_auto_compat.py
compat_config.py
compat_trace.py
low_latency_reference.py
```

确认 `deep_ep` 从待测构建导入。三个 backend 使用相同测试工具版本、相同输入参数和匹配硬件，各自使用新的输出目录。

## 无设备预览与主机检查

`--help` 和 `--dry-run` 不导入 Torch、torch_npu 或 deep_ep，不初始化 NPU：

```bash
python tests/ep/test_ep_auto_compat.py --dry-run \
  --backend candidate-auto --bs 2 64 65 256 2 \
  --cpu-sync false --capacity 256 --capacity-mode both \
  --barrier-in-prologue both --alignment 1 16 --edge-routes off
```

计划中每项有稳定 `case_id`、路由 seed、逐 rank B、统一 M 及所有功能开关。`--case-id ID...` 可在同一计划参数下精确复跑；选择不同 CPU 同步模式不会把其他相同用例重新编号。重复 BS 保留独立 occurrence ID，支持“小 batch→大 batch→小 batch”的复用测试。

主机测试：

```bash
cd tests/ep
python -B -m unittest test_compat_config test_compat_trace test_ep_auto_compat_host test_summarize_compat
cd ../comm
python -B -m unittest test_stream_host
```

planner、结果状态机和 profiler 解析测试只需标准库。合成 tensor 检查需要 CPU Torch；没有 Torch 时会显式跳过这组检查。这些通过不代表 NPU 内核已通过。

## 关键选项

| 选项 | 行为 |
| --- | --- |
| `--bs` | 默认 `2 32 33 64 65 128 256 2`；覆盖 EP128 的 B=32/33 MTE 边界，以及较大 BS 的 URMA 回退 |
| `--capacity` | 长期复用 buffer 的分配容量，默认 256；不随每次 B 缩小 |
| `--capacity-mode fixed/batch/both` | fixed 每次传 M=capacity；batch 每次传 M=max(各 rank B)；两者都复用同一容量的 buffer |
| `--cpu-sync false/true/default/all` | 默认 false；default 传 None 以执行 API 的默认行为。cached 调用始终 False，且不传累计 stats |
| `--barrier-in-prologue false/true/both` | 默认 false，覆盖运行时默认位置；设置发生在相关 EP 操作前 |
| `--alignment 1 16 128` | 选择需要的专家对齐值，默认 1/16 |
| `--deferred false/true/both` | 默认 both，所有操作开启有效异步事件；deferred 检查 hook 只执行一次 |
| `--deterministic false/true/both` | 默认 both，分别创建对应 buffer |
| `--route random hotspot` | 默认两种；随机参考可复现 |
| `--dtype bf16 fp8` | 默认 BF16；FP8 使用原生 INT16 scale packs，之后 combine 为 BF16 |
| `--scale-layout row col` | FP8 输入和返回 scales 的行/列布局；BF16 无 scales |
| `--edge-routes auto/off/only` | edge 使用逐 rank B=(0,2,128,32) 循环，含 masked/duplicate；M 为统一容量，不能各 rank 分别传各自 B |
| `--profile-paths` | 每个操作采集真实设备 kernel 证据；默认关闭并明确标记 unverified |
| `--output-dir` | 每 rank 独占创建日志，已有文件拒绝覆盖；三个 backend 和每次复跑使用不同目录 |

测试固定 H7168/K8/E128，生产规模为 EP8/16/32/64/128。默认 native 路径只有 EP128 的合格小 BS 允许选择 MTE；其他 EP（包括 EP1/2/4 诊断规模）预期 URMA。保留其他 EP 的测试，以验证官方路径的兼容性。FP8 测试数据要求实际 B≤256，BF16 要求 B≤4096，以保证 payload 身份字段可精确表示；这是本测试数据的限制，不是库的容量限制。M 可以更大，以验证容量导致的回退。

MTE 路径验收使用 EP128、arch3510、64 AIV、可用 UB 至少 190 KiB 的目标机器，并要求 `1≤B≤32`、`B≤M≤256`。BF16 dispatch 与 BF16 combine 按各自原始 B/M 计算预期；FP8 dispatch 预期 URMA，后续 BF16 combine 在上述条件满足时可预期 MTE。EP128 大 BS 保持 URMA，不因整组性能通过而强制 MTE。trace 中 UB、M、cached、CPU sync、barrier、dtype/scale pack 和权重分支均与用例核对。实际不符会失败，不以“正确性通过”掩盖意外回退。

## 设备运行示例

启动约定与已有测试一致：`WORLD_SIZE` 是节点数，`RANK` 是节点编号，`--num-processes` 是本节点 NPU 数。每个节点运行同一命令，使用同一 MASTER_ADDR/MASTER_PORT 和一致用例配置。不要再套 torchrun。

按照现有测试环境规则确认所用设备空闲后再运行。以下只是入口示例，T02 不自动启动这些作业。

先跑小型默认模式正确性，分别选择三个 backend 和对应已安装构建：

```bash
python tests/ep/test_ep_auto_compat.py \
  --backend candidate-auto --bs 2 64 65 256 2 \
  --capacity 256 --capacity-mode both --cpu-sync false \
  --barrier-in-prologue false --alignment 1 \
  --deferred false --deterministic false --route random \
  --edge-routes off --output-dir /tmp/ep-compat-auto-default-01
```

验证 EP128 的 MTE/URMA 边界时，将 BS 列表改成 `2 32 33 256 2`。其他 EP 的相同用例用于检查 URMA 回退。后续增加 alignment、deferred、deterministic 和 hotspot；默认 planner 不把 CPU sync=True 混进主矩阵，避免它截断其他用例。

CPU 同步单独核验官方和候选：

```bash
python tests/ep/test_ep_auto_compat.py \
  --backend official --expected-source-commit 3b25377d04b24fc6154698ded78a2bcb2c59afff \
  --bs 2 --cpu-sync true --alignment 1 --deferred false \
  --deterministic false --route random --edge-routes off \
  --output-dir /tmp/ep-compat-official-cpusync-01
```

同配置改成候选 backend 和对应期望 commit 后单独运行。默认 None 行为另用 `--cpu-sync default`。官方通过而候选失败才归为候选回归；双方失败需定位共同环境/基线问题，不能记作通过。

用小矩阵采集路径证据：

```bash
python tests/ep/test_ep_auto_compat.py \
  --backend candidate-auto --bs 2 --capacity 256 --capacity-mode both \
  --dtype bf16 fp8 --scale-layout row col --cpu-sync false \
  --alignment 1 --deferred false --deterministic false \
  --route random --edge-routes off --profile-paths \
  --output-dir /tmp/ep-compat-auto-paths-01
```

上述入口仍由启动环境决定全局 EP；单机 EP8 的新策略预期全部 URMA。验证 MTE 必须使用全局 EP128，不能只改 `--num-processes` 为 128。再单独使用 `--edge-routes only` 验证空 rank/不同 B，EP128 下还覆盖混合传输。保留不同路径的设备证据，不按 shape 猜测结果。

FP8 payload 和非单位 scales 用于检查通信字节、stride、padding 及缓存更新；合成专家函数用可精确表示的 FP8 值生成 BF16 输出。该用例不模拟完整模型的反量化/GEMM 性能。combine 同时检查调用者已乘权和未乘权的专家输出，权重回传不能再次改变 hidden。

## 事件与共享通信层

运行前必须确认目标运行时提供可调用的 `torch.npu._sleep`。此前检查的
torch_npu 2.12 环境缺少该接口；入口会显式注册 torch_npu，并在分布式初始化前
拒绝缺少延迟接口的环境。该环境仍需适配并验证设备端异步延迟，不能用 CPU sleep
或同步操作代替后宣称完成跨流验收。

保留原 H1024/K1 回退用例、EventOverlap 与 Bucket 冒烟，以及 H7168/K8/E128 shape 的 producer/consumer 双流用例。CLI 中的 `mte` shape 名称保持兼容；是否可用 MTE 还取决于全局 EP 和当前策略，以下单机 EP8 示例只验证 URMA 路径：

```bash
python tests/comm/test_stream.py --num-processes 8 \
  --ep-shape both --ep-bs 16 --ep-capacity 256 \
  --backend candidate-auto --barrier-in-prologue false
```

该入口检查延迟生产、提前释放输入、分配压力、dispatch 与 combine 的跨流等待，以及 immediate/deferred 两种完成方式。`--skip-bucket` 只用于隔离问题，完整公共层冒烟保留默认 Bucket 检查。不同 backend、barrier=True 另跑。

该测试不使用 profiler 证明 transport，避免 profiler 的同步影响流行为。路径证据由相同 shape 的主矩阵另行提供；标准输出显示真实包路径和配置。PP/Engram 等其他共享层回归仍使用原有专项测试，不因 Bucket 通过就宣称全部通过。

## 结果与汇总

每个 rank 的 `rank-00000.jsonl` 等文件包含：

- run 开始、计划 case ID，以及正常完成标记；
- 每个 case 的 started/pass/fail、输入配置和失败栈；
- 每个操作的预期传输与实际证据状态，cached/weights 等分支；
- 构建身份及工具哈希。

`--profile-paths` 还保存原始 profiler 输出及 `evidence.json`，包含真实设备 kernel 名、模板参数、文件哈希。解析只接受设备事件或 kernel CSV；CPU label、只有 epilogue、缺失主 kernel、混合或不匹配路径均不能通过。

将所有节点的逐 rank 日志收集到同一个目录，再汇总：

```bash
python tests/ep/summarize_compat.py --input-dir /tmp/ep-compat-auto-default-01
python tests/ep/summarize_compat.py --input-dir /tmp/ep-compat-auto-paths-01 --require-paths
```

缺 rank、缺 case、失败、截断 JSON、缺少正常结束标记或身份/计划冲突均不能得到完整通过。未开启 profiler 时，正确性可以通过，但路径状态保持 unverified；加 `--require-paths` 后这种结果不满足退出成功条件。

汇总依据 JSONL 中记录的设备证据及其文件哈希，不重新访问原节点的 trace 文件。交接时还须归档原始 profiler 目录；这份汇总不能替代对原始文件的独立审计。

测试中断后使用新的输出目录重新运行，或先从 dry-run 获取 case ID，再用相同配置筛选复跑；不要追加到不完整的旧日志，也不要删除 failed 行来制造通过结果。
