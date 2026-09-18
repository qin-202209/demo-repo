# 逻辑回归前向传播的向量化实现

本仓库对应《深度学习笔记》**第 57 页**的内容：用向量化的方式一次性计算 `m` 个训练样本的前向传播，去掉样本维度上的 `for` 循环。

```
Z = np.dot(w.T, X) + b      # b 通过广播扩展成 (1, m)
A = σ(Z)                    # σ 逐元素作用在 Z 上，一次得到所有 m 个样本的预测
```

其中：

| 变量 | 含义 | 形状 |
| --- | --- | --- |
| `X` | 训练样本（按列堆叠） | `(n_x, m)` |
| `w` | 权重列向量 | `(n_x, 1)` |
| `b` | 偏置（标量，广播） | `(1, 1)` |
| `Z` / `A` | 线性输出 / 激活输出 | `(1, m)` |

本仓库同时给出了 **NumPy 纯手写**和 **PyTorch** 两种实现，并附带**合成数据集**与**公开数据集**加载脚本。

## 目录结构

```
vectorized-nn/
├── numpy_version/
│   ├── sigmoid.py        # 数值稳定的 sigmoid
│   ├── forward.py        # 循环版 + 向量化版前向传播
│   └── demo.py           # 对比两种实现，展示广播形状
├── torch_version/
│   ├── model.py          # 张量实现 + nn.Linear 模块实现
│   └── demo.py           # 与 NumPy 结果对比
├── data/
│   ├── make_synthetic.py # 生成合成二分类数据集 (.npz / .csv)
│   └── load_public.py    # breast_cancer / digits / MNIST(0 vs 1)
├── tests/
│   └── test_vectorization.py
├── conftest.py
├── requirements.txt
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python numpy_version/demo.py    # 循环 vs 向量化
python torch_version/demo.py    # PyTorch vs NumPy
pytest -q                       # 等价性与稳定性测试
```

## 数据集

### 合成数据

```bash
python data/make_synthetic.py
```

会在 `data/generated/` 下生成：

* `synthetic.npz`：包含 `X (n_x, m)`、`y (1, m)`、`w_true (n_x, 1)`、`b_true (1, 1)`
* `synthetic.csv`：每行一个样本，列为 `feature_0, ..., label`

也可以在代码中直接调用：

```python
from data.make_synthetic import make_synthetic
data = make_synthetic(n_x=2, m=200, seed=0)   # 线性可分的高斯双月/双团数据
```

### 公开数据

```python
from data.load_public import load_breast_cancer, load_digits, load_mnist_binary

load_breast_cancer()            # 离线，30 维，二分类
load_digits(class_a=0, class_b=1)  # 离线，8x8 -> 64 维
load_mnist_binary()             # 首次需联网下载，784 维，0 vs 1
```

所有加载器统一返回 `{"X": (n_x, m), "y": (1, m)}`，可直接送入本仓库的前向传播函数。

## 测试覆盖

* 循环版与向量化版结果一致
* NumPy 与 PyTorch（含 `nn.Linear` 模块）结果一致
* sigmoid 在 `±1000` 等极端输入下不溢出
* 偏置广播行为正确
* 合成数据集形状与线性可分性
* 公开数据集加载器输出形状

## 参考

* 《深度学习笔记》第 57 页，吴恩达深度学习课程 · 第二周：神经网络的编程基础
* 关键词：向量化、广播（broadcasting）、逻辑回归前向传播
