# Kaggle House Prices：OOF 融合方案

这个目录实现 Kaggle 经典比赛 **House Prices: Advanced Regression Techniques** 的完整训练流程：

```text
官方 train/test
    ↓
数据审计
    ↓
fold 内缺失值处理、特征工程、偏态变换和编码
    ↓
Lasso / ElasticNet / GradientBoosting / XGBoost / LightGBM / CatBoost
    ↓
OOF 交叉验证
    ↓
weighted blending 或 Ridge stacking
    ↓
submission.csv
```

比赛目标在代码中使用 `log1p(SalePrice)` 训练和验证，提交前统一执行 `expm1`。

## 目录结构

```text
method_v1/
├─ configs/
│  └─ base.yaml
├─ scripts/
│  ├─ 01_audit_data.py
│  ├─ 02_train_oof.py
│  ├─ 03_ensemble.py
│  └─ 04_make_submission.py
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ features.py
│  ├─ preprocessing.py
│  ├─ models.py
│  ├─ cv.py
│  └─ ensembling.py
├─ outputs/                 # 首次运行时自动创建
├─ README.md
└─ requirements.txt
```

原始比赛数据保持在：

```text
26ML_program/house-prices-advanced-regression-techniques/
├─ train.csv
├─ test.csv
├─ sample_submission.csv
└─ data_description.txt
```

不要修改或覆盖这些原始文件。

## 环境安装

建议使用 Python 3.11 或 3.12。在目标运行环境的项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\method_v1\requirements.txt
```

`requirements.txt` 同时包含 XGBoost、LightGBM 和 CatBoost。模型工厂采用延迟导入；如果只安装部分模型依赖，训练脚本会跳过缺失的可选模型。

## 配置

默认配置位于 `configs/base.yaml`，包括：

- 输入和输出路径；
- 目标列和 ID 列；
- KFold 折数、随机种子；
- XGBoost、LightGBM 和 CatBoost 的 fold 内 early stopping；
- 特征工程开关；
- 面积、房龄、翻修状态和顺序类别特征；
- 偏态阈值及强制类别字段；
- 启用的模型和模型超参数。

相对路径以 `method_v1` 为基准。默认输出到 `method_v1/outputs`。

为了保证不同模型的 OOF 结果可融合，不要在模型之间随意修改 `cv.n_splits`、`cv.shuffle` 或 `cv.random_state`。

提升树默认启用 fold 内 early stopping：

```yaml
cv:
  early_stopping: true
  early_stopping_rounds: 100
```

每一折的预处理器只在该折训练部分拟合，转换后的验证部分仅用于选择最佳迭代数。`outputs/metrics/cv_scores.csv` 会记录每折的 `best_iteration`。Lasso、ElasticNet 和 sklearn GradientBoosting 不使用这一配置。

## 运行流程

下面的命令都从 `26ML_program` 项目根目录执行。

### 1. 数据审计

```powershell
python .\method_v1\scripts\01_audit_data.py --config .\method_v1\configs\base.yaml
```

生成：

```text
outputs/reports/dataset_summary.csv
outputs/reports/missing_values.csv
outputs/reports/categorical_levels.csv
outputs/reports/target_summary.json
```

这一步只检查数据，不生成全量清洗后的 CSV。

### 2. 训练基础模型并生成 OOF

训练配置中启用的全部模型：

```powershell
python .\method_v1\scripts\02_train_oof.py --config .\method_v1\configs\base.yaml
```

也可以先训练少量模型验证流程：

```powershell
python .\method_v1\scripts\02_train_oof.py `
  --config .\method_v1\configs\base.yaml `
  --models elasticnet gradient_boosting
```

主要输出：

```text
outputs/oof/<model>.csv
outputs/test_predictions/<model>.csv
outputs/metrics/cv_scores.csv
```

预测列名为 `prediction_log`，表示对数尺度预测，不能直接作为最终 `SalePrice` 提交。

默认不保存每折模型。如果确实需要复用已训练模型，可以添加 `--save-models`；模型将写入 `outputs/models`，文件体积可能较大。

### 3. 模型融合

默认使用非负且权重之和为 1 的 OOF 优化权重：

```powershell
python .\method_v1\scripts\03_ensemble.py `
  --config .\method_v1\configs\base.yaml `
  --method optimized
```

其他融合方式：

```powershell
# 等权平均
python .\method_v1\scripts\03_ensemble.py --method equal

# Ridge 元模型 stacking
python .\method_v1\scripts\03_ensemble.py --method stacking
```

如果未指定 `--models`，脚本只使用配置中已经存在 OOF 和测试预测文件的模型。显式指定模型时，缺少预测文件会直接报错。

主要输出：

```text
outputs/ensemble/oof_blend.csv
outputs/ensemble/test_blend.csv
outputs/ensemble/blend_metadata.json
outputs/ensemble/prediction_correlation.csv
outputs/ensemble/residual_correlation.csv
outputs/metrics/ensemble_scores.csv
```

### 4. 生成 Kaggle 提交文件

```powershell
python .\method_v1\scripts\04_make_submission.py `
  --config .\method_v1\configs\base.yaml
```

默认读取 `outputs/ensemble/test_blend.csv`，执行 `expm1`，按照官方 `sample_submission.csv` 的 ID 顺序生成：

```text
outputs/submissions/submission.csv
```

也可以指定预测文件和输出名称：

```powershell
python .\method_v1\scripts\04_make_submission.py `
  --prediction-file .\method_v1\outputs\test_predictions\lightgbm.csv `
  --output-name submission_lightgbm.csv
```

## 防止数据泄漏

项目遵循以下约束：

- 中位数、众数、偏态字段选择、标准化和 One-Hot 类别集合只在当前训练 fold 上拟合；
- 验证 fold 和测试集只调用已经拟合的转换器；
- 特征工程只使用当前行已有信息，不读取 `SalePrice`；
- CatBoost 保留类别列，不使用全量 One-Hot 数据；
- stacking 只使用基础模型的 OOF 预测作为二层特征；
- 最终融合依据 OOF 预测完成，不使用 Kaggle public leaderboard 反向调权。

## 模型预处理分支

| 模型 | 预处理 |
|---|---|
| Lasso / ElasticNet | 语义缺失值、偏态变换、标准化、One-Hot |
| GradientBoosting / XGBoost / LightGBM | 语义缺失值、中位数填充、One-Hot |
| CatBoost | 语义缺失值、数值中位数填充、保留类别字段 |

特征工程默认将房屋质量、地下室完成度、车库完成度等顺序类别映射为数值，例如 `Ex > Gd > TA > Fa > Po`。配置项如下：

```yaml
features:
  add_ratio_features: true
  add_quality_area_features: true
  add_ordinal_features: true
  drop_original_ordinal: true
```

比例特征包括地下室装修比例、地上面积比例、每房间居住面积、每卧室居住面积、每车位车库面积、卫浴卧室比、土地利用率和门廊面积比例。分母为 0 时比例记为 0；真正的数据缺失仍保留给 fold 内插补。

质量交互包括 `OverallGrade`、`QualLivArea`、`QualTotalSF`，以及外部质量、厨房质量、地下室质量、车库质量、壁炉质量、供暖质量和泳池质量与对应面积或数量的乘积。该组特征依赖 ordinal 特征，因此启用 `add_quality_area_features` 时应同时启用 `add_ordinal_features`。

如需与原始 One-Hot 方案进行消融对比，可以关闭 `add_ordinal_features`，并为该实验设置新的 `paths.output_dir`，避免覆盖原有 OOF 文件。

`None` 类缺失和普通未知缺失会被区分。例如没有地下室、车库或泳池属于房屋结构信息，不应简单当作随机缺失值处理。

## 实验建议

第一次运行时，建议先执行数据审计，然后只训练两个模型确认整个流程：

```powershell
python .\method_v1\scripts\02_train_oof.py `
  --config .\method_v1\configs\base.yaml `
  --models elasticnet gradient_boosting
```

确认 OOF、融合和 submission 文件都正常后，再训练 XGBoost、LightGBM 和 CatBoost。判断模型是否值得加入融合时，应同时查看 OOF RMSE 和残差相关性，而不是只看单模型排名。
