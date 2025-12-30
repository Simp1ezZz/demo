# LangChain 1.2+ Demo - HuggingFace 本地模型版本

这是一个 LangChain 1.2+ 版本的演示项目，使用 **HuggingFace 本地模型**，**无需 API Key**！

所有模型都在本地运行，完全免费使用。

## 功能演示

本项目包含以下演示：

1. **基本的 LLM 调用** - 展示如何使用 LangChain 调用大语言模型
2. **提示模板** - 演示如何使用提示模板格式化输入
3. **链式调用** - 展示 LLMChain 的使用方法
4. **多步骤处理** - 演示如何将多个步骤串联起来
5. **对话式交互** - 展示如何维护对话上下文

## 🚀 快速开始

### 方式一：Google Colab（推荐）

**最简单的方式！** 直接在 Google Colab 中打开并运行：

1. 打开 `langchain_demo.ipynb` 文件
2. 上传到 Google Colab 或直接在 Colab 中打开
3. 按照 notebook 中的步骤依次运行各个 cell
4. **无需 API Key！** 模型会自动从 HuggingFace 下载

**优势：**
- ✅ 无需 API Key
- ✅ 无需本地安装环境
- ✅ 可以直接运行和修改
- ✅ 适合学习和实验
- ✅ 完全免费使用

### 方式二：本地运行

## 安装步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**注意：** 首次安装可能需要一些时间，因为需要安装 PyTorch 和 Transformers。

### 2. 运行演示

```bash
python main.py
```

**首次运行说明：**
- 程序会自动从 HuggingFace 下载模型（默认使用 `Qwen/Qwen2.5-0.5B-Instruct`）
- 下载时间取决于网络速度（模型约 1GB）
- 模型会缓存在本地，后续运行会更快
- 无需任何 API Key 或配置！

## 项目结构

```
demo/
├── langchain_demo.ipynb  # Google Colab 版本（推荐）
├── main.py              # 本地运行版本
├── requirements.txt     # Python 依赖
├── README.md           # 项目说明
└── .env                # 环境变量配置（需要自己创建）
```

## 依赖说明

- `langchain>=1.2.0`: LangChain 核心库（1.2+ 版本）
- `langchain-community>=0.2.0`: 社区贡献的组件（包含 HuggingFace 集成）
- `langchain-core>=0.2.0`: LangChain 核心组件
- `transformers>=4.35.0`: HuggingFace Transformers 库
- `torch>=2.0.0`: PyTorch（深度学习框架）
- `accelerate>=0.24.0`: HuggingFace Accelerate（加速推理）
- `sentencepiece>=0.1.99`: 分词器支持

## 模型选择

默认使用 `Qwen/Qwen2.5-0.5B-Instruct`（小模型，速度快）。你可以在代码中修改 `MODEL_NAME` 变量来使用其他模型：

**推荐模型：**
- `Qwen/Qwen2.5-0.5B-Instruct` - 小模型，速度快，适合快速测试
- `Qwen/Qwen2.5-1.5B-Instruct` - 中等大小，平衡性能和速度
- `THUDM/chatglm3-6b` - 中文效果好，但需要更多内存

**修改方法：**
- 在 `main.py` 中修改 `MODEL_NAME` 变量
- 在 `langchain_demo.ipynb` 的第二个代码 cell 中修改 `MODEL_NAME` 变量

## 注意事项

1. **首次运行**：需要下载模型，可能需要一些时间（取决于网络速度）
2. **内存要求**：根据模型大小，需要 2GB-16GB 内存
3. **GPU 支持**：如果有 GPU，会自动使用 GPU 加速（更快）
4. **磁盘空间**：模型会缓存在 `~/.cache/huggingface/` 目录
5. **网络连接**：首次运行需要网络连接以下载模型

## 扩展建议

你可以基于这个 demo 进一步探索：

- 添加向量数据库集成
- 使用 LangChain 的工具和代理功能
- 实现文档问答系统
- 构建自定义链和代理

## 参考资源

- [LangChain 官方文档](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)

