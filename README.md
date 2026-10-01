# 仲景问典 · 开源核心（local-rag-kit + 中医经典数据集）

> 中医经典 AI 学习工具的**本地 RAG 开源实现**：把《伤寒论》《金匮要略》《素问》《难经》《神农本草经》
> 共 2024 条原文装进本地检索，配你自己配置的大模型（本地 Ollama / DeepSeek / 通义 等），
> 以「仲景」口吻讲经方、查条文、览方剂。**零商业依赖、不内置任何模型与密钥。**
>
> 🌐 在线体验（GitHub Pages）：https://andrew2king.github.io/zhongjing-wendian

## 开源了什么 / 没开源什么

| | 开源（本仓库，MIT） | 闭源收费（完整版） |
|---|---|---|
| 数据集 | ✅ 五经原文 2024 条 + 经方 25 条 | — |
| 检索框架 | ✅ bigram TF-IDF + 可选语义混合检索 | — |
| RAG 后端 | ✅ `local_rag.py`（检索 + LLM + 合规护栏） | — |
| 前端 demo | ✅ `web/index.html` 三页签 | — |
| **开箱即用** | ❌ | ✅ Windows 双击 exe、一键本机运行 |
| **多厂商后台** | ❌ | ✅ 10 家厂商预设 + 密钥管理 |
| **更新支持** | ❌ | ✅ 持续维护 |

**完整版（开箱即用 Windows 版 / 源码包）即将上架闲鱼，搜索「仲景问典」即可购得。**
开源是为了让更多人能本地、免费地把中医经典用起来；完整版卖的是「省心 + 服务」。

## 目录结构

```
仲景问典-open-source/
├── corpus/                 # 中医经典原文数据集（开源）
│   ├── books.json          #   五经原文 2024 条
│   ├── formulas.json       #   经方 25 条
│   └── README.md
├── rag-kit/                # 本地 RAG 框架（开源）
│   ├── local_rag.py        #   后端：检索 + 大模型 + 合规护栏
│   ├── requirements.txt
│   └── README.md
├── web/                    # 前端 demo（开源）
│   └── index.html          #   医圣对话 / 条文速查 / 方剂速查
├── LICENSE                 # MIT + 医疗合规声明
└── README.md
```

## 快速开始

```bash
cd rag-kit
pip install -r requirements.txt

# 1) 仅检索（无需任何 Key）
python local_rag.py
# 浏览器打开 http://127.0.0.1:8000  → 条文/方剂速查直接可用

# 2) 云端大模型（如 DeepSeek）
LLM_API_KEY=sk-xxxx LLM_MODEL=deepseek-v4-flash python local_rag.py

# 3) 本地 Ollama（零成本、完全离线）
ollama pull qwen3.8:latest && ollama pull bge-m3
LLM_API_TYPE=ollama LLM_BASE=http://localhost:11434/v1 LLM_MODEL=qwen3.8:latest \
EMBED_ENABLED=1 EMBED_BASE_URL=http://localhost:11434 EMBED_MODEL=bge-m3 \
python local_rag.py
```

详见 `rag-kit/README.md`。

## 合规红线（贯穿始终）

- 本工具**仅用于中医经典学习**，不诊断、不开方、不推荐具体用药。
- 输出侧已做正则护栏：命中个性化医疗建议一律拦截改写，并强制附免责声明。
- 数据集为作者逝世逾百年的公版古籍，结构化整理以 MIT 开放。

## License

[MIT](LICENSE) · 附带医疗合规声明（见 LICENSE 末尾）。

## 致谢

- 伤寒论 / 金匮要略 / 素问 / 难经：`wangekxy/classical-tcm-canon`
- 神农本草经（孙星衍辑本）：`xiaopangxia/TCM-Ancient-Books`
- 由「硅基起源」整理与开源。
