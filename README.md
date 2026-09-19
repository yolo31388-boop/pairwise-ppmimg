# ppmimg — 纯标准库的 PPM 图像处理小库

零第三方依赖（仅 Python 标准库），支持 PPM（P3 文本 / P6 二进制）读写和常用滤镜。
所有滤镜函数都**返回新 Image，不修改原图**。

## 环境要求

- Python 3.8+（运行库本身不需要任何第三方包）
- 跑测试需要 `pytest`

## 快速上手

```python
from ppmimg import (
    read_ppm, write_ppm,
    to_grayscale, invert, adjust_brightness, adjust_contrast,
    gaussian_blur, sharpen, sobel_edge, resize, rotate,
)

img = read_ppm("input.ppm")            # P3 / P6 自动识别

gray   = to_grayscale(img)             # 灰度
neg    = invert(img)                   # 反色
bright = adjust_brightness(img, 1.2)   # 亮度 x1.2
cont   = adjust_contrast(img, 0.8)     # 对比度 x0.8
blur   = gaussian_blur(img, 3)         # 高斯模糊，半径 1-5
sharp  = sharpen(img)                  # 锐化
edges  = sobel_edge(img)               # Sobel 边缘检测（输出灰度图）
small  = resize(img, 400, 300)         # 双线性缩放
rot    = rotate(img, 90)               # 顺时针 90/180/270

write_ppm(blur, "out_p6.ppm", binary=True)    # 存 P6（默认）
write_ppm(blur, "out_p3.ppm", binary=False)   # 存 P3
```

## Image 类

```python
from ppmimg import Image

img = Image(4, 4)                       # 新建 4x4 黑图
img = Image(2, 1, [(255, 0, 0), (0, 255, 0)])  # 用像素列表构造
img.width, img.height                   # 尺寸
img.pixels                              # 扁平 [(r, g, b), ...]，行优先
r, g, b = img.get_pixel(0, 0)           # 读像素（越界抛 IndexError）
img.set_pixel(1, 0, (10, 20, 30))       # 写像素（自动截断到 0-255）
```

## 批量处理整个目录

自带命令行脚本 `batch_filter.py`：

```bash
python batch_filter.py 输入目录/ 输出目录/ grayscale
python batch_filter.py 输入目录/ 输出目录/ blur --radius 3
python batch_filter.py 输入目录/ 输出目录/ brightness --factor 1.2
python batch_filter.py 输入目录/ 输出目录/ resize --width 400 --height 300
python batch_filter.py 输入目录/ 输出目录/ rotate --degrees 90
python batch_filter.py 输入目录/ 输出目录/ sobel --p3   # 输出 P3 文本格式
```

## 实现细节（便于核对结果）

- **灰度**：BT.601 亮度公式 `round(0.299R + 0.587G + 0.114B)`，例如纯红 `(255,0,0)` → `76`。
- **反色**：每通道 `255 - v`。
- **亮度**：每通道 `v * factor`，截断到 0-255。
- **对比度**：每通道 `(v - 128) * factor + 128`，截断到 0-255（128 为不动点）。
- **高斯模糊**：`sigma = radius / 2`，核大小 `2*radius+1`，可分离两次一维卷积（水平 + 垂直），中间结果用浮点保精度。
- **锐化**：3x3 核 `[0,-1,0, -1,5,-1, 0,-1,0]`。
- **Sobel**：先转灰度，再算 Gx/Gy，幅值 `sqrt(Gx^2 + Gy^2)` 截断到 255，输出灰度图。
- **缩放**：双线性插值，目标像素中心对齐源图：`src = (dst + 0.5) * (src_size / dst_size) - 0.5`。
- **旋转**：顺时针 90/180/270 度。
- **边界处理**：所有卷积类操作（模糊/锐化/Sobel）对越界坐标按**边缘像素复制**（clamp）处理；缩放同样 clamp 到边缘，1x1 图也不会崩。
- **PPM 解析**：自动跳过 `#` 注释；读取时支持 maxval 非 255（会按比例映射到 0-255），写出固定 maxval=255。

## 运行测试

```bash
python -m pytest
```

测试覆盖：P3/P6 往返与互转、灰度公式、反色、亮度/对比度、纯色模糊不变与边界中间值、
Sobel 平坦区为 0 / 垂直边缘高响应、2x2→4x4 双线性手算核对、旋转、
以及 800x600 图上 `gaussian_blur(3)` 和 `resize` 各自 < 10 秒的性能用例。

## 目录结构

```
ppmimg/
    __init__.py     # 包入口，导出全部公开 API
    image.py        # Image 类
    io.py           # PPM P3/P6 读写
    filters.py      # 全部滤镜函数
tests/
    test_ppmimg.py  # pytest 测试
batch_filter.py     # 批量处理命令行脚本
pytest.ini
```
