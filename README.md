# NeuralCleanse 🧠🔍

**NeuralCleanse** 是一个用于检测和逆向工程神经网络后门攻击的工具。本项目实现了对 MNIST 数据集的后门攻击，并提供了检测和逆向工程触发器的功能。复现自论文 [NeuralCleanse](https://ieeexplore.ieee.org/document/8835365)。

## 🚀 项目简介

本项目旨在帮助研究人员和安全专家理解神经网络后门攻击的机制，并提供一种有效的方法来检测和逆向工程这些攻击。通过以下主要功能，NeuralCleanse 为您提供了一套完整的工具链：

- 🛠️ 预训练一个带有后门的神经网络模型
- 🔬 逆向工程触发器和掩码
- 🕵️‍♂️ 检测模型中的异常标签

## 📥 安装

### 1. 克隆仓库

首先，将本仓库克隆到本地：

```sh
git clone https://github.com/yourusername/NeuralCleanse.git
cd NeuralCleanse
```

### 2. 安装依赖

安装所需的 Python 包：

```sh
pip install -r requirements.txt
```

## 🛠️ 使用方法

### 预训练带有后门的模型

运行 `pretrain.py` 脚本来训练一个带有后门的模型：

```sh
python pretrain.py
```

### 逆向工程触发器和掩码

运行 `rev_eng.py` 脚本来逆向工程触发器和掩码：

```sh
python rev_eng.py
```

### 检测异常标签

运行 `ano_det.py` 脚本来检测模型中的异常标签：

```sh
python ano_det.py
```

## 📂 目录结构

```
NeuralCleanse/
├── pretrain.py        # 预训练带有后门的模型
├── rev_eng.py         # 逆向工程触发器和掩码
├── ano_det.py         # 检测模型中的异常标签
├── model.py           # 定义了神经网络模型
└── dataload.py        # 用于加载 MNIST 数据集
```

## 🤝 贡献

欢迎贡献代码！请 fork 本仓库并提交 pull request。我们非常期待您的贡献，无论是功能改进、bug 修复还是文档更新。

<!-- ## 📜 许可证

本项目采用 **MIT 许可证**。详情请参见 [LICENSE](LICENSE) 文件。 -->
