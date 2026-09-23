# local-rag-kit · 仲景问典本地 RAG 框架（开源）

从「仲景问典」完整产品抽取的最小可运行后端，**零商业依赖、不内置任何模型/密钥**，把
「古籍原文检索 + 本地/云端大模型讲解」的核心能力开放出来，供开发者自行部署与二次开发。

## 能力

- **关键词检索**（bigram TF-IDF）—— 纯算法，无需任何 API Key，开箱即用
- **可选语义检索** —— 接入本地 Ollama（bge-m3）或任意 OpenAI 兼容 embedding
- **大模型对话** —— OpenAI 兼容接口 / 本地 Ollama，模型由你自己配置
- **输出侧合规护栏** —— 正则拦截个性化医疗建议（开方/荐药），强制附免责声明

## 快速开始

```bash
pip install -r requirements.txt

# 1) 不配大模型：检索直接可用
python local_rag.py
# → http://127.0.0.1:8000  （此时 /api/chat 会提示先配置模型）

# 2) 云端模型（如 DeepSeek）
LLM_API_KEY=sk-xxxx LLM_MODEL=deepseek-v4-flash python local_rag.py

# 3) 本地 Ollama（零成本、离线）
ollama pull qwen3.8:latest            # 对话模型
ollama pull bge-m3                     # 语义检索（可选）
LLM_API_TYPE=ollama LLM_BASE=http://localhost:11434/v1 LLM_MODEL=qwen3.8:latest \
EMBED_ENABLED=1 EMBED_BASE_URL=http://localhost:11434 EMBED_MODEL=bge-m3 \
python local_rag.py
```

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 运行状态、模型/embedding 是否就绪、检索模式 |
| GET | `/api/search?q=太阳病&k=5&book=伤寒论` | 混合检索（语义就绪=hybrid，否则 keyword） |
| POST | `/api/chat` | body `{messages, context}`，仲景口吻讲解，带合规护栏 |
| GET | `/api/data/books` `/api/data/formulas` | 原始数据集 |

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | 云端 OpenAI 兼容 base；Ollama 填 `http://localhost:11434/v1` |
| `LLM_API_KEY` | 空 | 云端模型 Key；Ollama 可留空 |
| `LLM_MODEL` | `deepseek-v4-flash` | 模型名 |
| `LLM_API_TYPE` | `openai` | `openai` / `ollama` |
| `EMBED_ENABLED` | `0` | `1` 开启语义检索 |
| `EMBED_BASE_URL` | `http://localhost:11434` | embedding 基址 |
| `EMBED_MODEL` | `bge-m3` | embedding 模型 |
| `SEARCH_ALPHA` | `0.5` | 混合检索中语义权重（0=纯关键词，1=纯语义） |

## 目录

```
local_rag.py     # 后端（检索 + LLM + 合规护栏 + 静态托管 web/）
requirements.txt
README.md
```

数据集在仓库根 `../corpus/books.json`。完整产品（开箱即用 Windows 版、多厂商后台、一键本机运行）
见根 README 的「完整版」说明。
