# 模块与扩展

数据流：元数据/配对 → 冻结输入 → OCR → 文字处理 → 几何处理 → 指标 → 配对/页面组分析。

| 层 | 输入/输出 | 依赖边界 |
|---|---|---|
| data | ImageRecord / PairTask / LogicalPageGroup | 只读图片和JSON；不加载模型或数据目录中的Python脚本 |
| transforms | RGB数组、文字框 → RGB数组、Mask、审计信息 | 不认识Eleme/WeChat/模拟器的评价结果 |
| metrics | 图片/页面组 → 特征或指标、证据 | 明确接收模型目录、设备或API客户端 |
| execution | 实验配置 → 可续跑工作目录 | 组织阶段、模型进程和状态，不实现评分公式 |
| analysis | 冻结任务与结果 → JSON、Markdown、图 | 不调用模型或API，不将失败补零 |
| storage | 路径、流式SHA256、JSON/JSONL | 不包含业务配置 |

图像编码、响应解析和评分实现位于 `metrics`。评分协议的来源和许可证记录在 `third_party`。

## 扩展点

- OCR：在 `transforms.detectors.DETECTORS` 注册工厂，接收配置、显式缓存路径和输入样本，返回检测函数及权重指纹。
- 文字处理：在 `transforms.text.TEXT_TRANSFORMS` 注册函数，接收RGB、框、参数，返回图像/共同Mask/审计信息。渲染层检查Mask一致、Mask外像素不变。
- 几何：在 `transforms.preprocessing.PREPROCESSORS` 注册规格构建函数和图像处理函数；输入输出尺寸必须明确记录。
- 编码器：在 `metrics.encoders.ENCODERS` 注册实例工厂，实现 `feature(path,spec)` 和独立双图 `cosine(a,b,spec)`。向models.lock.json加入本地权重和哈希，并在实验metrics中选择。
- 数据集：新增元数据适配器，转换到data/types.py的契约；已有中文元数据由registry.py读取。分组语义由数据配对/组清单表达，不交给编码器推测。

默认配置包含5种图像条件，后续新增名称只在对应注册表与配置中声明。扩展实现随包安装，不从数据目录加载代码。

## 配置与缓存

JSON中的project_root相对于配置文件，paths相对于project_root；显式绝对路径也支持。CLI的project覆盖项目根目录。API环境变量优先于项目.env，没有隐式读取GUI-CC配置。

每个run固定配置。特征以图片ID/条件命名，记录图像SHA256、模型/预处理/精度/库版本和硬件协议、输入张量SHA。模型参数或输入不匹配时拒绝续写；移动整个项目不改变协议中的根路径。实现版本与代码来源单独追溯。

分析使用同一方案的共同成功样本；跨几何使用两几何×选定模型×选定条件的共同成功交集。提供逐对均值及共同页面、页面组等权统计。均分上升不等同于指标更准确。自动错误页压力对照不是人工金标准。

前台启动时为本次实验分配运行 ID，模型进程直接向当前终端输出进度。分析读取所选运行 ID 内的评分。

逻辑页面组配置支持group_manifest：省略时由配对构建，填写时读取项目相对或绝对路径的人工组JSONL，沿用rafe_logical_page_group_v1并验证环境、图片哈希、状态和证据标签。缺失组不计零分，部分可比较结果保留null和needs_review状态。
