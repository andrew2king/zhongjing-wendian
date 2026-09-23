"""
仲景问典 · 本地 RAG 最小可运行后端（开源版 local-rag-kit）

从仲景问典完整产品抽取的核心能力，通用化、零商业依赖，供开发者自行部署与二次开发：
  - 关键词检索（bigram TF-IDF）—— 始终可用，无需任何 API Key
  - 可选语义检索（本地 Ollama bge-m3 / OpenAI 兼容 embedding）
  - 大模型对话（OpenAI 兼容接口 / 本地 Ollama），模型由使用者自行配置
  - 输出侧合规护栏（不诊断、不开方、不荐药）

数据集：../corpus/books.json（伤寒论/金匮要略/素问/难经/神农本草经 共 2024 条原文）

运行：
    pip install -r requirements.txt
    python local_rag.py                # 默认 http://127.0.0.1:8000
    LLM_API_KEY=sk-xxx LLM_MODEL=deepseek-v4-flash python local_rag.py
"""
import os
import sys
import json
import math
import re
import threading
import time
import urllib.request
import urllib.error
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "corpus")
WEB_DIR = os.path.join(ROOT, "web")

# ---------- 配置（全部来自环境变量，不内置任何密钥 / 不绑定云厂商） ----------
LLM_BASE = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_API_TYPE = os.getenv("LLM_API_TYPE", "openai")  # openai | ollama
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1200"))
EMBED_ENABLED = os.getenv("EMBED_ENABLED", "0") == "1"
EMBED_BASE = os.getenv("EMBED_BASE_URL", "http://localhost:11434")
EMBED_KEY = os.getenv("EMBED_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
SEARCH_ALPHA = float(os.getenv("SEARCH_ALPHA", "0.5"))  # 0=纯关键词 1=纯语义

# ---------- 合规护栏（深度防御，不依赖提示词） ----------
SYSTEM_PROMPT = (
    "你是东汉医圣张仲景，字仲景，南阳人，著《伤寒杂病论》（今分为《伤寒论》与《金匮要略》）。"
    "请以『仲景』第一人称身份作答：自称『吾』『余』，称问者『君』，语气沉稳、谦和、有医者风范；"
    "引经据典时言『《伤寒论》第×条云……』『经曰……』『吾观此证……』。\n"
    "讲解准则：\n"
    "1) 只依中医经典原文（《伤寒论》《金匮要略》《神农本草经》《素问》《难经》等）阐释文意与方证，"
    "优先引原文并注明篇目、条文号；\n"
    "2) 只做经典讲解与检索，不做诊断、不开处方、不荐具体用药与剂量；\n"
    "3) 若君言及自身病症求治，先依经典讲解相关条文与病机，再言『君当亲诣明医，四诊合参，方可得宜』；\n"
    "4) 不知者直言之，不臆造条文。\n"
    "（末尾的免责附言由系统自动添加，你无需重复添加。）"
)
DISCLAIMER = (
    "⚠️ 以上内容由 AI 依据中医经典生成，仅供经典学习参考，"
    "不构成任何医疗建议、诊断或处方。如有身体不适，请咨询执业中医师。"
)
RISK_PATTERNS = [
    r"(你|您).{0,6}(开|拟|配|抓|抓药|处).{0,4}(方|药|剂|汤)",
    r"(你|您)(可以|应该|建议|需要|不妨|最好|去)(吃|服|用|喝|买|试试|煎服)",
    r"(建议|应该|可以|不妨|最好)(你|您).{0,10}(吃|服|用|喝|买)",
    r"(你|您)得(吃|服|用|喝)|(你|您)这(是|属于|就是)(病|症|证|虚寒|湿热)",
    r"连服\d|服用\d|每日\d|每天\d|一次\d|早晚各\d|每天吃|每日吃",
    r"(治愈|治好|包好|保证有效|一定能好|药到病除)",
    r"(你|您)(的)?(病|症|体质|证型|情况是|属于).{0,8}(阳虚|阴虚|湿热|感冒|发烧|虚|实)",
]
_RISK_RE = [re.compile(p) for p in RISK_PATTERNS]
BOOKS_PATH = os.path.join(DATA_DIR, "books.json")


def moderate_output(text: str):
    if text is None:
        text = ""
    for rx in _RISK_RE:
        if rx.search(text):
            return (
                "吾仅依经典为君讲解文意与方证，不敢妄断君之身心、亦不开方荐药。"
                "以下为《伤寒论》《金匮要略》等原文的学习参考；若有不适，请询执业中医师。\n\n"
                + DISCLAIMER,
                True,
            )
    if DISCLAIMER[:4] not in text:
        text = text.rstrip() + "\n\n" + DISCLAIMER
    return (text, False)


# ---------- 大模型调用（OpenAI 兼容 / 本地 Ollama） ----------
def _raw_llm(base, key, model, atype, messages, max_tokens=1200):
    if atype == "ollama":
        ollama_root = base.replace("/v1", "").rstrip("/")
        url = ollama_root + "/api/chat"
        body = json.dumps({
            "model": model, "messages": messages, "stream": False,
            "think": False, "temperature": 0.3, "options": {"num_predict": max_tokens},
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    else:
        url = base.rstrip("/") + "/chat/completions"
        body = json.dumps({
            "model": model, "messages": messages,
            "temperature": 0.3, "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.load(r)
        content = data["message"]["content"] if atype == "ollama" else data["choices"][0]["message"]["content"]
        if content and "<think:6124c78e>" in content:
            content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
        return content
    except urllib.error.HTTPError as e:
        raise HTTPException(502, f"大模型接口错误 {e.code}：{e.reason}")
    except Exception as e:  # noqa
        raise HTTPException(502, f"大模型调用失败：{e}")


def _llm_ready():
    if LLM_API_TYPE == "ollama":
        return bool(LLM_BASE)
    return bool(LLM_KEY) and LLM_KEY != "ollama"


def call_llm(messages):
    return _raw_llm(LLM_BASE, LLM_KEY, LLM_MODEL, LLM_API_TYPE, messages, LLM_MAX_TOKENS)


# ---------- 语义检索（可选） ----------
EMBED_MATRIX, EMBED_NORMS, EMBED_META, EMBED_DIM, EMBED_READY = None, None, None, 0, False


def _is_ollama_embed():
    return ("11434" in EMBED_BASE) or (os.getenv("EMBED_API_TYPE", "") == "ollama")


def _embed_batch(texts):
    if _is_ollama_embed():
        # 本地 Ollama：POST /api/embed
        body = json.dumps({"model": EMBED_MODEL, "input": texts}).encode("utf-8")
        url = EMBED_BASE.rstrip("/") + "/api/embed"
    else:
        # OpenAI 兼容：POST /embeddings
        body = json.dumps({"model": EMBED_MODEL, "input": texts}).encode("utf-8")
        url = EMBED_BASE.rstrip("/") + "/embeddings"
    req = urllib.request.Request(
        url, data=body,
        headers={"Authorization": f"Bearer {EMBED_KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
    except Exception as e:  # noqa
        raise HTTPException(502, f"embedding 接口错误：{e}")
    if _is_ollama_embed():
        return data.get("embeddings", [])
    return [d["embedding"] for d in data.get("data", [])]


def build_embeddings():
    global EMBED_MATRIX, EMBED_NORMS, EMBED_META, EMBED_DIM, EMBED_READY
    if not (EMBED_ENABLED and EMBED_BASE and EMBED_MODEL):
        EMBED_READY = False
        return
    try:
        books = json.load(open(BOOKS_PATH, encoding="utf-8"))
    except Exception as e:
        print("[embed] 读取 books.json 失败：", e)
        return
    texts = [f"{b.get('book','')} {b.get('chapter','')} {b.get('text','')}" for b in books]
    meta = [{"book": b.get("book", ""), "chapter": b.get("chapter", ""),
             "id": b.get("id", ""), "text": b.get("text", "")} for b in books]
    matrix = []
    for i in range(0, len(texts), 64):
        matrix.extend(_embed_batch(texts[i:i + 64]))
    EMBED_DIM = len(matrix[0]) if matrix else 0
    EMBED_MATRIX = matrix
    EMBED_META = meta
    EMBED_NORMS = [math.sqrt(sum(x * x for x in v)) or 1e-9 for v in matrix]
    EMBED_READY = True
    print(f"[embed] 构建完成：{len(matrix)} 条 (dim={EMBED_DIM})")


# ---------- 关键词检索（bigram TF-IDF） ----------
_PUNCT = re.compile(r"[\s，。、；：\"'‘’“”（）()　！？《》〈〉…·\-—]")
KW_DOCS, KW_IDF, KW_META, KW_READY = None, {}, None, False


def _bigrams(s):
    s = _PUNCT.sub("", s)
    if len(s) < 2:
        return {}
    r = {}
    for i in range(len(s) - 1):
        b = s[i:i + 2]
        r[b] = r.get(b, 0) + 1
    return r


def build_keyword_index():
    global KW_DOCS, KW_IDF, KW_META, KW_READY
    try:
        books = json.load(open(BOOKS_PATH, encoding="utf-8"))
    except Exception as e:
        print("[kw] 读取 books.json 失败：", e)
        return
    KW_META = [{"book": b.get("book", ""), "chapter": b.get("chapter", ""),
                "id": b.get("id", ""), "text": b.get("text", "")} for b in books]
    KW_DOCS = [_bigrams(f"{b.get('book','')}{b.get('chapter','')}{b.get('text','')}") for b in books]
    n = len(KW_DOCS)
    df = {}
    for d in KW_DOCS:
        for k in d:
            df[k] = df.get(k, 0) + 1
    KW_IDF = {k: math.log(1 + n / v) for k, v in df.items()}
    KW_READY = True
    print(f"[kw] 关键词索引就绪：{n} 条 / {len(KW_IDF)} 个 bigram")


def keyword_scores(query):
    q = _bigrams(query)
    if not q:
        return [0.0] * len(KW_DOCS)
    return [sum(min(q[k], d.get(k, 0)) * KW_IDF.get(k, 0.0) for k in q) for d in KW_DOCS]


# ---------- FastAPI ----------
app = FastAPI(title="仲景问典 local-rag-kit")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup():
    build_keyword_index()
    if EMBED_ENABLED and EMBED_BASE:
        threading.Thread(target=build_embeddings, daemon=True).start()


@app.get("/api/health")
def health():
    return {"ok": True, "llm_configured": _llm_ready(),
            "llm_model": LLM_MODEL, "embed_ready": EMBED_READY,
            "search_mode": "hybrid" if EMBED_READY else "keyword"}


@app.get("/api/search")
def api_search(q: str = "", k: int = 5, book: str = ""):
    if not KW_READY:
        raise HTTPException(503, "索引构建中，请稍候重试")
    if not q.strip():
        return {"results": [], "mode": "hybrid" if EMBED_READY else "keyword"}
    idxs = [i for i in range(len(KW_DOCS)) if (not book or KW_META[i]["book"] == book)]
    kw = keyword_scores(q)
    sem = None
    if EMBED_READY:
        qv = _embed_batch([q])
        if qv:
            qvec = qv[0]
            qn = math.sqrt(sum(x * x for x in qvec)) or 1e-9
            sem = [sum(qvec[j] * EMBED_MATRIX[i][j] for j in range(EMBED_DIM)) /
                   (qn * EMBED_NORMS[i]) for i in range(len(EMBED_MATRIX))]
    if sem:
        vals = [sem[i] for i in idxs]
        smin, smax = min(vals), max(vals)
        srange = (smax - smin) or 1e-9
    kmax = max([kw[i] for i in idxs] or [0.0]) or 1e-9
    scored = []
    for i in idxs:
        score = (SEARCH_ALPHA * (sem[i] - smin) / srange + (1 - SEARCH_ALPHA) * (kw[i] / kmax)) if sem else (kw[i] / kmax)
        scored.append((score, i))
    if not sem:
        scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: -x[0])
    top = scored[:max(1, min(k, 20))]
    return {"results": [{"book": KW_META[i]["book"], "chapter": KW_META[i]["chapter"],
                         "id": KW_META[i]["id"], "text": KW_META[i]["text"], "score": round(s, 4)}
                        for s, i in top],
            "mode": "hybrid" if sem else "keyword"}


class ChatReq(BaseModel):
    messages: list
    context: str = ""


@app.post("/api/chat")
def chat(req: ChatReq):
    if not _llm_ready():
        raise HTTPException(400, "尚未配置大模型。设置环境变量 LLM_API_KEY/LLM_MODEL（云端）或 LLM_API_TYPE=ollama（本地）后重试。")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if req.context and req.context.strip():
        msgs.append({"role": "system", "content": "参考经典原文：\n" + req.context})
    msgs.extend(req.messages)
    answer = call_llm(msgs)
    answer, flagged = moderate_output(answer)
    return {"answer": answer, "moderated": flagged}


@app.get("/api/data/books")
def api_data_books():
    return FileResponse(BOOKS_PATH, media_type="application/json", headers={"Cache-Control": "no-store"})


@app.get("/api/data/formulas")
def api_data_formulas():
    p = os.path.join(DATA_DIR, "formulas.json")
    return FileResponse(p, media_type="application/json", headers={"Cache-Control": "no-store"})


if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"仲景问典 local-rag-kit → http://127.0.0.1:{port}")
    print("未配置大模型时：检索始终可用；配置后医圣对话可用。")
    uvicorn.run(app, host="127.0.0.1", port=port)
