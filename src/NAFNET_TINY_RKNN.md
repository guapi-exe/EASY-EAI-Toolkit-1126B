# NAFNet-tiny 转 RKNN 说明

当前代码已经支持在上传 face 图时优先尝试加载 `nafnet_tiny.rknn`：

- 模型路径：`nafnet_tiny.rknn`
- 加载位置：`UploaderTask`
- 加载失败时会自动回退到当前传统去模糊流程，不影响现有运行

## 当前接入约束

当前 `NAFNetTinyEnhancer` 已按当前模型输入输出适配：

- 输入尺寸：`256 x 256`
- 输入通道：`RGB`
- 输入数据：`float32`
- 输入范围：`0..1`
- 输入布局：自动适配 `NHWC / NCHW`
- 输出通道：3 通道 RGB
- 输出数据：`float`
- 输出范围：优先按 `0..1` 处理，超范围时按 `0..255` 处理
- 输出后处理：范围裁剪，不再对负值做整体平移

如果你现在没有 `.rknn`，需要先准备下面两样之一：

1. `nafnet_tiny.onnx`
2. `nafnet_tiny.pth`，然后先导出为 `onnx`

## 推荐转换规格

建议先按下面规格转换，和当前接入代码最匹配：

1. 目标平台：`rv1126b`
2. 输入尺寸：`256 x 256`
3. 输入通道：`RGB`
4. 输入类型：`float32`
5. 输入范围：`0..1`
6. 输出尺寸：与输入同尺寸
7. 输出通道：3 通道 RGB
8. 尽量不要使用额外的 `mean/std` 预处理配置

## 推荐量化方式

RV1126B 上建议先做两轮：

1. 先导出 `非量化` `.rknn`，确认输入输出正确
2. 再做 `INT8` 量化，校准集优先用现场人脸图

校准集建议覆盖：

1. 白天/夜晚
2. 近景/中景
3. 轻微模糊/快速运动模糊
4. 不同肤色和背景亮度

## 转换步骤

### 1. 在 PC 上安装 RKNN Toolkit2

建议在 Ubuntu 或官方支持的 Python 环境里操作，不要在 RV1126B 板端转换。

### 2. 准备 ONNX

如果手里只有 `pth`，先导出成 `onnx`。

推荐导出规格：

1. 固定输入尺寸 `1x3x256x256`
2. opset 建议 `11` 或 `12`
3. 尽量避免动态 shape

### 3. 非量化导出

参考脚本：

`src/tools/convert_nafnet_tiny_to_rknn.py`

```bash
python src/tools/convert_nafnet_tiny_to_rknn.py \
  --onnx nafnet_tiny.onnx \
  --output nafnet_tiny.rknn \
  --target rv1126b
```

### 4. INT8 量化导出

例如：

```bash
python src/tools/convert_nafnet_tiny_to_rknn.py \
  --onnx nafnet_tiny.onnx \
  --output nafnet_tiny_int8.rknn \
  --dataset dataset/nafnet_face_calib.txt \
  --target rv1126b \
  --quant
```

### 5. 把生成的文件放到程序运行目录

文件名保持为：

`nafnet_tiny.rknn`

当前代码会自动尝试加载。

## dataset 文件格式

`--dataset` 需要传一个文本文件，每行一张图片路径，例如：

```txt
dataset/nafnet_face_calib/img001.jpg
dataset/nafnet_face_calib/img002.jpg
dataset/nafnet_face_calib/img003.jpg
```

## 运行后看什么日志

加载成功：

```txt
NAFNetTinyEnhancer: loaded nafnet_tiny.rknn input=256x256x3 input_fmt=1 input_type=0 output_fmt=1
```

模型不存在：

```txt
NAFNetTinyEnhancer: model file not found or unreadable: nafnet_tiny.rknn
```

增强成功：

```txt
UploaderTask: face enhanced by NAFNet-tiny
```

## 当前接入限制

当前版本只在 `face` 上传链路尝试使用 NAFNet-tiny，`person` 图仍走传统增强。这样做是为了：

1. 降低 NPU 压力
2. 优先把最关键的人脸上传效果做好
3. 避免实时主流程被增强模型拖慢
