# 验证记录

2026-10-03。本地实现与静态/合成验证完成。

已执行只读参考源码审查、旧pilot摘要及重启审计检查、权重文件SHA256核对。

未执行checkpoint加载、真实Repair、GPU测试、benchmark、states、真实标签采集、实验模型训练、pilot、完整搜索或依赖安装。

本地测试使用合成状态、合成标签、模拟Repair失败和临时账本。抽取原版VRP类执行Destroy/端点结构方法，不涉及torch模型路径。合成模型fit不属于真实开发标签训练。

## 实际执行与结果

- 系统 Python 3.11.1 执行 `python -B point_select_cli.py check` 和 `budget`，退出码0；该系统环境没有NumPy或PyTorch。只读检查仍可运行，checkpoint hash正确，默认总上限71936且所有caps为null。
- Codex已提供的运行时 Python 3.12.14 / NumPy 2.3.5 执行 `python -B -m unittest discover -s tests -p 'test_*.py' -v`：**21项全部通过**，最终一次运行3.441秒。未安装任何依赖；此版本不是服务器要求的NumPy1.26.4，不能宣称已验收指定服务器软件组合。
- 本地合成覆盖：原版Point/Destroy/端点语义，快审计特征一致性，20个非depot删除，重复槽位权重与共享标签，A/B流分离，逐rollout z和零/稀疏事件，实例级四折和折内预处理，已知合成排序恢复，零标签与随机平局，未批准caps和policy阻断，未--start无文件副作用，started/failed/重复任务不可覆盖，完成产物篡改检测，预测先于测试标签及A参照不读B。
- 完整合成集成：8实例/4开发，状态1/2步、3槽位、A2/B3，用模拟后端贯通独立benchmark、states、候选、特征、开发A、60次合成fit、测试预测、测试A参照、测试B和分析。测试验证阶段摘要、逐实例结果、结构消融、校准登记及测试后未登记benchmark拒绝。所有合成报告明确标注synthetic，不能推荐科学意义的完整搜索扩展。
- audit/analyze默认只读：集成测试比较执行前后run内全部文件大小与mtime，完全一致；不打开SQLite/WAL。
- 已核对复制的actor.py、repair.py、vrp/vrp_problem.py与参考项目文件SHA256逐项一致；checkpoint SHA256为指定值。运行核心未修改。
- 首轮合成验证发现并修复常数评分ndarray的JSON序列化问题。Python3.12 Windows tempfile的用户ACL与沙箱令牌不兼容；测试辅助改为项目内普通权限临时目录，并验证清理路径边界。首次留下的5个目录已在授权范围内清理，最终临时区仅保留.gitkeep。

## 尚未验证的边界

无torch环境，没有checkpoint反序列化、真实actor forward、贪心Repair连接、CUDA确定性/架构、真实显存和速度验收。服务器必须预配NumPy==1.26.4和适配GPU的PyTorch；真实CLI写入阶段会强制版本与显式启动/预算，加载器检查V100 sm_70且无CPU回退。

没有真实开发标签训练、pilot结果或完整搜索结果。12实例历史排重清单不代表全历史覆盖。人工重启/损坏SQLite恢复需要独立审计，本版没有自动重启或恢复命令。硬中断无export时只读审计明确报告未封存及不确定上限，不能把它当作零成本。
