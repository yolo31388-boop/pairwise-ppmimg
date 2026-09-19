# ppmimg

一个纯 Python 标准库实现的小型 PPM 图像处理库，不依赖 PIL / OpenCV / numpy 等第三方库。

- 支持 PPM 的 `P3`（ASCII 文本）和 `P6`（二进制）两种格式的读取与保存
- 提供 `Image` 类（宽、高、RGB 像素数据，`get_pixel` / `set_pixel`）
- 滤镜全部返回新图片，不修改原图
- 空间类滤镜（模糊、锐化、Sobel）边界采用**边缘像素复制**（edge clamp），越界不崩
- 800x600 图上 `gaussian_blur(3)` 约 3 秒、`resize` 约 1.5 秒（纯 Python 实测）

## 目录结构

```
ppmimg/
├── __init__.py    # 包入口，导出公共 API
├── image.py       # Image 类
├── ppm.py         # P3/P6 解析与保存
├── filters.py     # 全部滤镜
└── __main__.py    # 批量处理命令行（python -m ppmimg）
tests/
└── test_ppmimg.py # pytest 测试（51 个）
```

## 安装 / 运行环境

无需安装依赖，把 `ppmimg/` 目录放到项目里直接 import 即可。运行时只使用标准库；
`pytest` 仅在跑测试时需要。

## 快速上手

```python
from ppmimg import ppm, filters

img = ppm.load("input.ppm")          # 自动识别 P3 / P6

gray = filters.to_grayscale(img)     # 每种滤镜都返回新 Image
edge = filters.sobel_edge(gray)

ppm.save(edge, "out_p3.ppm", fmt="P3")   # 保存为文本格式
ppm.save(edge, "out_p6.ppm", fmt="P6")   # 保存为二进制格式（默认）
```

`Image` 上也挂了同名方法，链式调用更顺手：

```python
from ppmimg import Image

img = Image.load("input.ppm")
out = (
    img
    .adjust_brightness(1.2)
    .gaussian_blur(radius=3)
    .resize(640, 480)
    .rotate(90)
)
out.save("output.ppm")                # 默认 P6
```

## 直接构造图片

```python
from ppmimg import Image

# 像素按行优先排列：[第0行..., 第1行..., ...]，每个像素是 (r, g, b)
pixels = [
    (255, 0, 0), (0, 255, 0),
    (0, 0, 255), (255, 255, 0),
]
img = Image(2, 2, pixels)

print(img.width, img.height)          # 2 2
print(img.get_pixel(1, 0))            # (0, 255, 0)
img.set_pixel(0, 1, (10, 20, 30))
```

内存中与字节串互转（方便测试 / 网络传输）：

```python
data_p6 = ppm.dumps(img, "P6")
data_p3 = ppm.dumps(img, "P3")
img2 = ppm.loads(data_p6)
assert img2 == img
```

## 滤镜一览

所有滤镜都在 `ppmimg.filters` 下，且不修改输入图：

| 函数 | 说明 |
| --- | --- |
| `to_grayscale(img)` | Rec. 601 灰度：`0.299R + 0.587G + 0.114B`，结果 R=G=B |
| `invert(img)` | 反色：`255 - v`，255 变 0、0 变 255 |
| `adjust_brightness(img, factor)` | 乘法调亮：`v * factor`，factor<1 变暗、>1 变亮，自动 clamp |
| `adjust_contrast(img, factor)` | 以 128 为中心：`128 + (v-128)*factor`，0 变全灰、1 不变 |
| `gaussian_blur(img, radius)` | 可分离高斯模糊，`radius` 取 1-5；sigma 默认 `radius/2`（下限 0.5） |
| `sharpen(img)` | 中心 5、四邻域 -1 的 3x3 锐化核 |
| `sobel_edge(img)` | Sobel 边缘检测，输出灰度图，`clamp(hypot(gx, gy), 0, 255)`，边缘亮 |
| `resize(img, width, height)` | 双线性插值，角点严格对齐 |
| `rotate(img, degrees)` | 只接受 `90 / 180 / 270`（顺时针） |

边界处理：`gaussian_blur`、`sharpen`、`sobel_edge` 在访问越界像素时统一
取最近的合法像素（边缘复制），因此 1 像素宽/高的图也能正常处理。

## 关于数值约定（验收时可对照）

- 灰度：`round(0.299*R + 0.587*G + 0.114*B)`，所以纯红 (255,0,0) 灰度值是 `76`。
- 亮度 / 对比度结果均为 `round(...)` 后 clamp 到 `[0, 255]`。
- resize 双线性采样坐标为 `sx = dx * (src_w - 1) / (dst_w - 1)`（y 同理），
  即 2x2 放大到 4x4 时采样位置是 `0, 1/3, 2/3, 1`，角点像素不变。
- 高斯权重使用 1024 定点整数（两遍可分离卷积，每遍向下取整后再除一次），
  纯色图模糊后像素值严格不变。
- PPM 写出时 `maxval` 固定为 255（8 位）；读取时支持任意 1-65535 的 maxval
  （含 16 位 P6），会自动线性缩放到 0-255。

## 命令行批量处理

```bash
# 把当前目录所有 ppm 先模糊(radius=3)再转灰度，输出到 out/
python -m ppmimg *.ppm -o out/ -f blur:3 -f grayscale

# 缩放并强制输出为 P3 文本格式
python -m ppmimg a.ppm -o done/ -f resize:320x240 --format P3

# 旋转 + 边缘检测
python -m ppmimg *.ppm -o edge/ -f rotate:90 -f sobel
```

`-f/--filter` 可重复，按出现顺序串联执行。支持的滤镜写法：

```
grayscale | invert | sharpen | sobel
brightness:<系数>     contrast:<系数>
blur:[半径]           resize:<宽>x<高>
rotate:<90|180|270>
```

## 运行测试

```bash
pytest
```

> Windows 下如果 pytest 的默认临时目录（`%TEMP%\pytest-of-*`）没有访问权限，
> 可以把临时目录指定到工作区：
> `pytest --basetemp=.pytest-tmp`

测试覆盖：P3/P6 往返互转、注释与畸形文件、灰度公式、反色、亮度/对比度、
模糊（纯色不变 / 黑白交界出现中间值 / 各半径 / 边界）、锐化、Sobel（平坦区为 0、
垂直边缘饱和为 255）、resize 手算对照、旋转三个方向，以及 800x600 的性能上限。
