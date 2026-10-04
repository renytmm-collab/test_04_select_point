# 固定 Repair 的原生 Point 选择

独立 Linux/Windows 实验包。本地只完成静态/合成验证；真实权重测试、测速、采集及训练需在服务器显式启动。没有训练Repair、联合学习或完整NLNS搜索入口。

协议见 `protocol/POINT_SELECTION_PROTOCOL.md`，已知历史排重仅覆盖12实例。官方权重已随包复制。上传整个目录，保留相对布局；旧项目和旧结果不需要上传。不要把输出目录设为指向项目外的符号链接。

## 环境与只读检查

Python>=3.10，NumPy==1.26.4；PyTorch>=2.5由用户预先配置，V100要求wheel含sm_70。没有命令自动安装依赖。`requirements.txt`只声明NumPy。无需scikit-learn，Ridge和小型加权CART使用NumPy、保存JSON参数，不使用pickle模型。

真实CLI写入阶段严格检查NumPy==1.26.4；后续审计/分析要求与run冻结的NumPy版本一致。模型加载前设置确定性cuBLAS工作区并禁用CUDA驱动磁盘缓存；没有静默CPU回退。合成验证的本地NumPy版本见验证记录，不能替代服务器指定版本验收。

在服务器项目根目录执行下列命令。推荐缓存禁用入口 `python -B point_select_cli.py`（即使省略-B，此脚本也在导入包前禁用bytecode）。不要直接导入真实backend进行环境检查。

```bash
python -B point_select_cli.py check
python -B point_select_cli.py budget
```

check仅读取安装元数据、源码AST、配置和权重文件哈希，不导入torch/numpy、不加载权重、不调用Repair。GPU架构只能在显式真实阶段核对；不把check当作GPU验证。

配置模板为 `configs/pilot.template.json`。在首次真实执行前自行复制为 `configs/pilot.json` 并确定设备、seed与规模；运行后不要改冻结配置/源码/协议/历史清单/权重。默认新root_seed=2026100304，32实例/24开发/8测试，50/200步、16槽位、A32/B128。全新实验版本应使用新seed和新目录。

caps默认null，模板规模不批准计算。每个真实阶段的显式 `--start --cap N` 才提交该阶段预算，收据记录上限；配置caps非null时CLI不能超过它。

## 命令、写入和真实预算

|命令|写入|真实Repair|
|---|---|---|
|check / budget|无|0|
|audit|无|0|
|benchmark|独立run的manifest、stage receipt/ledger/export|单独预算，含预热|
|states|研究run的manifest、catalog/split/folds、快照、state账本|上限6400|
|candidates / features|阶段槽位、唯一集合、特征、审计、receipt|0|
|collect-dev-a|开发A阶段账本|上限24576|
|fit|CV、预处理、四个模型及主模型冻结|0 Repair，60次fit|
|freeze-test-predictions|测试选择、分数、在线计时和决策政策|0|
|collect-test-a|测试A账本|上限8192|
|freeze-test-reference|仅A产生的昂贵选择参照|0|
|collect-test-b|独立测试B账本|上限32768|
|analyze|默认无，显式--output才写不可变报告|0|

所有写入阶段需要--start。没有已完成阶段自动重跑；started/failed/歧义阶段阻断。成功阶段重复也拒绝，使用audit查看。账本含SQLite/WAL和export；只读audit不打开SQLite。

## 分阶段服务器运行

下列每条都是独立启动指令，含真实预算的命令应逐条审阅后手动执行，不编成一键脚本。示例配置需由你先准备。

独立初解测速；batch1预热2/稳态5为7次，batch16为112次，均在71936之外：

```bash
python -B point_select_cli.py benchmark --config configs/pilot.json --run runs/bench-single-001 --start --batch-size 1 --cap 7
python -B point_select_cli.py benchmark --config configs/pilot.json --run runs/bench-batch-001 --start --batch-size 16 --cap 112
```

初解测速不保证代表演化后状态。states完成后可指定 `--snapshot runs/pilot-001/stages/states/snapshot-I-STEP.json` 在新的benchmark目录测量。建议使用预先选定开发实例第50/200步快照；测试结果不得用于挑选最有利测速。

用于D工程门槛的benchmark必须匹配一个已冻结开发状态，并在freeze-test-predictions时通过 `--benchmark-run` 登记。初解测速只供预算和描述性比值，不能单独通过D。

研究状态、候选、特征：

```bash
python -B point_select_cli.py states --config configs/pilot.json --run runs/pilot-001 --start --cap 6400
python -B point_select_cli.py candidates --config configs/pilot.json --run runs/pilot-001 --start
python -B point_select_cli.py features --config configs/pilot.json --run runs/pilot-001 --start
python -B point_select_cli.py collect-dev-a --config configs/pilot.json --run runs/pilot-001 --start --cap 24576
python -B point_select_cli.py fit --config configs/pilot.json --run runs/pilot-001 --start --max-fits 60
```

fit只消费开发A。先查看开发CV和benchmark，再自行准备 `configs/decision.policy.json`：复制policy模板，确定min_gain_pp（百分点）和max_online_repair_ratio（无量纲），不能留null。政策不改run冻结配置，在测试预测阶段完整复制并冻结。不得B后修改。

```bash
python -B point_select_cli.py freeze-test-predictions --config configs/pilot.json --run runs/pilot-001 --start --policy configs/decision.policy.json --benchmark-run runs/bench-single-001
python -B point_select_cli.py collect-test-a --config configs/pilot.json --run runs/pilot-001 --start --cap 8192
python -B point_select_cli.py freeze-test-reference --config configs/pilot.json --run runs/pilot-001 --start
python -B point_select_cli.py collect-test-b --config configs/pilot.json --run runs/pilot-001 --start --cap 32768
```

B评价测试全部唯一候选，以原生槽位期望为主基线；常数、规则、四个模型和A参照全部共享同一份B标签。最大总Repair=71936，候选去重减少A/B实际调用，失败/独立测速另计。

只读审计和分析（分析会在内存生成报告并输出stdout）：

```bash
python -B point_select_cli.py audit --config configs/pilot.json --run runs/pilot-001
python -B point_select_cli.py analyze --config configs/pilot.json --run runs/pilot-001 --benchmark-run runs/bench-single-001
```

只有显式--output写报告，输出必须位于本项目中：

```bash
python -B point_select_cli.py analyze --config configs/pilot.json --run runs/pilot-001 --benchmark-run runs/bench-single-001 --output runs/pilot-001/report.json
```

不提供单Repair benchmark时，D报告校准不足；B批次摊销比仅作为吞吐量代理，不伪装成单次延迟。报告分别列出实际唯一成功事件和重复槽位加权事件、各阶段墙钟/Repair准备/训练成本。

如需D判据，把上面freeze/analyze命令的benchmark路径替换为开发快照的单次Repair测速目录；可重复 `--benchmark-run` 注册多个预先确定的状态。未在预测冻结前注册的benchmark拒绝用于最终分析，避免测试后挑校准结果。

## 本地合成验证与中断处理

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

测试会在 `tests/_artifacts/` 创建临时文件并清理，故它是写入命令。不会加载checkpoint或运行真实Repair。实际原版Destroy测试抽取源码中的VRP类，仅运行结构方法，未验证torch/Tensor/模型路径。运行环境证据见 `VALIDATION.md`。

发生硬中断：保留原run，运行只读audit；缺export表示账本未封存，不能当作零开销。不要删除stage、改seed、重分批或覆盖started记录。没有自动重启工具；需独立恢复审计和明确授权，记录原账本哈希、状态计数、已知/不确定调用、追加预算及新目录，最终报告附全部前驱成本。本版采用严格阻断，尚不提供SQLite损坏恢复或跨服务器账本合并。
