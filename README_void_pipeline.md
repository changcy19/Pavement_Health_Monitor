# 脱空预测流水线

## 1. Abaqus 导出 RPT

在 Abaqus 中运行 `export_batch_xy_from_odb.py`，它会扫描 `data/` 下所有 `*.odb`，按文件名中的工况生成 `rpt_data/` 下的 `*.rpt`。

## 2. RPT 转 CSV

命令行运行：

```bash
python convert_rpt_to_csv.py --input-dir rpt_data --output-dir csv_data
```

## 3. 构建训练数据

```bash
python build_void_dataset.py --data-dir data --csv-dir csv_data --output dataset_void.csv
```

## 4. 训练模型

```bash
python train_void_model.py --dataset dataset_void.csv --output-dir model_out
```

## 文件名标签规则

例如 `TG0_H22_E5_A_V60_M10.odb`：

- `TG0` 温度梯度 0
- `H22` 面层厚度 22 cm
- `E5` 面层弹性模量 5 GPa
- `A` 脱空宽度 5 cm
- `V60` 速度 60 km/h
- `M10` 轴重 10 t
