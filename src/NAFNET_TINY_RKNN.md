# NAFNet-tiny 转 RKNN 说明

当前代码已经支持在上传 face 图时优先尝试加载 `nafnet_tiny.rknn`：

- 模型路径：`nafnet_tiny.rknn`
- 加载位置：`UploaderTask`
- 加载失败时会自动回退到当前传统去模糊流程，不影响现有运行

如果你现在没有 `.rknn`，需要先准备下面两样之一：

1. `nafnet_tiny.onnx`
2. `nafnet_tiny.pth`，然后先导出为 `onnx`

## 推荐输入规格

建议先按下面规格转换，和当前接入代码最匹配：

1. 输入尺寸：`256 x 256`
2. 输入通道：`RGB`
3. 输入布局：`NHWC` 或 `NCHW` 都可以，当前代码会自动识别常见输出布局
4. 输出尺寸：与输入同尺寸
5. 输出通道：3 通道 RGB
6. 数据范围：
   - 最优：输出 `0..1` 的 float
   - 也可接受：输出 `0..255` 的 float

## 推荐量化方式

RV1126B 上建议优先尝试：

1. `INT8` 量化
2. 校准集使用你自己的现场人脸图
3. 校准图片数量建议先放 100 到 300 张

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

### 3. 准备校准集目录

例如：

`dataset/nafnet_face_calib/`

里面放若干 jpg/png 图。

### 4. 运行转换脚本

参考脚本：

`src/tools/convert_nafnet_tiny_to_rknn.py`

示例：

```bash
python src/tools/convert_nafnet_tiny_to_rknn.py \
  --onnx nafnet_tiny.onnx \
  --output nafnet_tiny.rknn \
  --dataset dataset/nafnet_face_calib.txt \
  --target rv1126
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
NAFNetTinyEnhancer: loaded nafnet_tiny.rknn input=256x256x3
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

## 下一步建议

如果你已经有 `onnx`，优先做这三件事：

1. 先转出一个 `float` 或 `非量化` 的 `.rknn`，确认输入输出方向和效果是对的
2. 再做 `INT8` 量化
3. 把运行日志和一张增强前后的 face 样图拿出来，看是否需要调整：
   - 输入归一化
   - 输出反归一化
   - 模型输入尺寸
