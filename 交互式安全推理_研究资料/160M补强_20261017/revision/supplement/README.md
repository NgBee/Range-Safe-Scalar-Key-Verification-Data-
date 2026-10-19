# 配套补充材料

当前论文的证据分为以下三份数据包，完整归档时应一并保存：

1. [SupplementaryServer160M.zip](SupplementaryServer160M.zip)：本轮新增的14M/31M/160M服务器请求、日志、编译图、源快照、初始化计时、离线复算与当前论文模板，并含10月19日十五值检查及发布构建来源证据。
2. [SupplementaryCodeAndData_20261017.zip](../../../审稿补强_20261017/reproducibility/packages/final_v3/SupplementaryCodeAndData_20261017.zip)：此前组件、生成、故障注入、缓存扩展、数值干预实验的代码与数据，包括保留的失败和排除记录。
3. [SupplementaryNumericalLogits_20261017.zip](../../../审稿补强_20261017/reproducibility/packages/final_v3/SupplementaryNumericalLogits_20261017.zip)：此前数值诊断的完整logit数组。

新增包保留用户提供的原始文件内容。统计以`revision/results/server_summary.json`为准；收到时的两份近似/历史summary同时保留，包内README说明其差别。离线复算只需要Python 3.11和NumPy 1.26.4；不需模型权重，也不会启动服务器执行脚本。

两个旧包保持不变。当前源包与新增服务器补充包的逐文件清单在各ZIP的`MANIFEST.json`内；整包校验和见[SHA256SUMS](../SHA256SUMS)，整包校验和见[SHA256SUMS](../SHA256SUMS)。

10月19日的理论更新已纳入两份新版ZIP。原始实验脚本按收到时的哈希固定；工作区中后续更新的run_generation_160.py与smoke_160m.py保留原状，打包使用original_server_scripts中的两份原始快照。
