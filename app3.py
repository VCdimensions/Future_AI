# app.py
import re
import json
import time
import os
import html
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from io import BytesIO


# -----------------------------
# Config / Spec
# -----------------------------
@dataclass
class SourceSpec:
    key: str
    name: str
    url: str
    mode: str  # "udn_html" | "cnyes_api" | "cnyes_keyword"
    start_heading: Optional[str] = None
    stop_heading_candidates: Optional[List[str]] = None
    stop_heading: Optional[str] = None
    cnyes_slug: Optional[str] = None
    keyword: Optional[str] = None
    pick_rule: str = "generic"


UDN_6644 = "https://udn.com/news/cate/2/6644"
UDN_6645 = "https://udn.com/news/cate/2/6645"

CNYES_WD_STOCK_PAGE = "https://news.cnyes.com/news/cat/wd_stock"
CNYES_SEARCH_US_AFTER = "https://www.cnyes.com/search/all?keyword=%E7%BE%8E%E8%82%A1%E7%9B%A4%E5%BE%8C"
CNYES_SEARCH_NY_FX = "https://www.cnyes.com/search/all?keyword=%E7%B4%90%E7%B4%84%E5%8C%AF%E5%B8%82"
CNYES_SEARCH_ENERGY = "https://www.cnyes.com/search/all?keyword=%E8%83%BD%E6%BA%90%E7%9B%A4%E5%BE%8C"
CNYES_SEARCH_METALS = "https://www.cnyes.com/search/all?keyword=%E8%B2%B4%E9%87%91%E5%B1%AC%E7%9B%A4%E5%BE%8C"

SOURCES: List[SourceSpec] = [
    # 1) UDN 6644 財經焦點
    SourceSpec(
        key="udn_6644_focus",
        name="UDN 產經 6644｜財經焦點（挑 2 則：影響台股最重要）",
        url=UDN_6644,
        mode="udn_html",
        start_heading="財經焦點",
        stop_heading_candidates=["金融要聞", "稅務法務", "產業綜合", "個人理財", "房地產"],
        pick_rule="tw_impact",
    ),
    # 2) UDN 6645 股市要聞
    SourceSpec(
        key="udn_6645_market",
        name="UDN 股市 6645｜股市要聞（挑 2 則：台股盤勢分析）",
        url=UDN_6645,
        mode="udn_html",
        start_heading="股市要聞",
        stop_heading_candidates=["存股族愛ETF", "上市電子", "店頭未上市", "國際財經"],
        pick_rule="tw_taiex_tape",
    ),
    # 3) UDN 6645 上市電子
    SourceSpec(
        key="udn_6645_elec",
        name="UDN 股市 6645｜上市電子（挑 2 則：重要個股）",
        url=UDN_6645,
        mode="udn_html",
        start_heading="上市電子",
        stop_heading_candidates=["店頭未上市", "國際財經", "熱門排行", "焦點股"],
        pick_rule="tw_single_stock",
    ),
    # 4-7) 鉅亨 search 頁：新聞第一則
    SourceSpec(
        key="cnyes_kw_us_after",
        name="鉅亨｜搜尋『美股盤後』新聞第一則",
        url=CNYES_SEARCH_US_AFTER,
        mode="cnyes_keyword",
        keyword="美股盤後",
        pick_rule="us_market",
    ),
    SourceSpec(
        key="cnyes_kw_ny_fx",
        name="鉅亨｜搜尋『紐約匯市』新聞第一則",
        url=CNYES_SEARCH_NY_FX,
        mode="cnyes_keyword",
        keyword="紐約匯市",
        pick_rule="fx_market",
    ),
    SourceSpec(
        key="cnyes_kw_energy",
        name="鉅亨｜搜尋『能源盤後』新聞第一則",
        url=CNYES_SEARCH_ENERGY,
        mode="cnyes_keyword",
        keyword="能源盤後",
        pick_rule="energy_market",
    ),
    SourceSpec(
        key="cnyes_kw_metals",
        name="鉅亨｜搜尋『貴金屬盤後』新聞第一則",
        url=CNYES_SEARCH_METALS,
        mode="cnyes_keyword",
        keyword="貴金屬盤後",
        pick_rule="metals_market",
    ),
    # 8) 鉅亨 category API
    SourceSpec(
        key="cnyes_wd_stock",
        name="鉅亨｜wd_stock（挑 2 則：美股盤勢分析）",
        url=CNYES_WD_STOCK_PAGE,
        mode="cnyes_api",
        cnyes_slug="wd_stock",
        pick_rule="us_market",
    ),
]

SOURCES_UI_ORDER = [
    "udn_6644_focus",
    "udn_6645_market",
    "udn_6645_elec",
    "cnyes_kw_us_after",
    "cnyes_kw_ny_fx",
    "cnyes_kw_energy",
    "cnyes_kw_metals",
    "cnyes_wd_stock",
]
SOURCES_BY_KEY = {s.key: s for s in SOURCES}
SOURCES_UI: List[SourceSpec] = [SOURCES_BY_KEY[k] for k in SOURCES_UI_ORDER]

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}

DEFAULT_GPT_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_GPT_BASE = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
DEFAULT_GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")


# -----------------------------
# HTTP helper (with timing)
# -----------------------------
def http_get(url: str, headers: dict, params: Optional[dict] = None, timeout: int = 20):
    t0 = time.perf_counter()
    r = requests.get(url, headers=headers, params=params, timeout=timeout, allow_redirects=True)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return r, elapsed_ms


# -----------------------------
# Generic helpers
# -----------------------------
def norm_url(base: str, href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return base.rstrip("/") + href
    return href


def clean_title(title: str) -> str:
    t = (title or "").strip()
    if not t:
        return ""
    t = t.replace("\\u003c", "<").replace("\\u003e", ">")
    t = html.unescape(t)
    if "<" in t and ">" in t:
        t = BeautifulSoup(t, "lxml").get_text(" ", strip=True)
    t = re.sub(r"\s+", " ", (t or "")).strip()
    return t


def dedup_keep_order(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for it in items:
        u = (it.get("url") or "").strip()
        t = clean_title(it.get("title") or "")
        if not u or not t:
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append({"title": t, "url": u})
    return out


# -----------------------------
# Scoring (auto-pick)
# -----------------------------
KW = {
    "tw_impact": [
        (r"台積電|TSMC|2330", 8),
        (r"外資|三大法人|賣超|買超|降評|升評", 6),
        (r"台股|加權|指數|期貨|台指", 6),
        (r"匯率|新台幣|美元|美債|殖利率", 5),
        (r"Fed|聯準會|降息|升息|利率|CPI|通膨|非農", 5),
        (r"AI|伺服器|半導體|晶圓|先進製程|CoWoS", 4),
        (r"地緣政治|制裁|戰|油價|能源", 3),
    ],
    "tw_taiex_tape": [
        (r"收盤|開盤|盤中|大盤|加權|成交量|量能", 8),
        (r"跌|漲|重挫|反彈|回測|失守|站回", 6),
        (r"法人|外資|投信|自營商", 5),
        (r"台積電|2330", 4),
        (r"月線|季線|年線|5日線|均線|技術面", 4),
    ],
    "tw_single_stock": [
        (r"（\d{4}）|\b\d{4}\b", 8),
        (r"營收|法說|財報|EPS|毛利|獲利|展望|下修|上修", 6),
        (r"漲停|跌停|大漲|大跌|重挫", 4),
        (r"台積電|聯發科|鴻海|聯電|日月光|廣達|緯創", 3),
    ],
    "us_market": [
        (r"美股|道瓊|那斯達克|標普|費半|四大指數", 8),
        (r"盤後|收盤|期指|期貨", 6),
        (r"殖利率|美元|通膨|CPI|非農|Fed|利率", 5),
        (r"Nvidia|輝達|Apple|特斯拉|微軟|Amazon|Meta|Google|Alphabet", 4),
    ],
    "fx_market": [
        (r"匯市|美元|日圓|歐元|英鎊|美債|殖利率", 8),
        (r"Fed|聯準會|利率|通膨|CPI|非農", 6),
    ],
    "energy_market": [
        (r"原油|WTI|Brent|油價|OPEC|庫存|EIA", 8),
        (r"能源盤後|盤後", 6),
    ],
    "metals_market": [
        (r"黃金|金價|白銀|貴金屬|美元|殖利率", 8),
        (r"盤後", 5),
    ],
    "generic": [(r".*", 1)],
}


def build_gpt_pick_prompt(
    spec: SourceSpec,
    items: List[Dict[str, str]],
    top_n: int,
    extra_prompt: str = "",
) -> Tuple[str, str]:
    lines = []
    for i, it in enumerate(items):
        title = clean_title(it.get("title") or "")
        lines.append(f"{i}. {title}")
    list_text = "\n".join(lines)
    system = "你是專業財經新聞編輯，負責從候選標題中挑選最相關、最重要的新聞。若提供額外指令，必須優先遵守。"
    extra = (extra_prompt or "").strip()
    extra_block = f"\n額外指令（最高優先，必須遵守）：{extra}\n" if extra else ""
    user = (
        f"區塊名稱：{spec.name}\n"
        f"請從下列候選標題中，挑選最符合此區塊目的的 {top_n} 則。\n"
        f"{extra_block}"
        "只回傳 JSON 陣列（0-based index），例如：[0,2]\n"
        "候選標題：\n"
        f"{list_text}"
    )
    return system, user


def extract_text_from_gpt_response(data: dict) -> str:
    if isinstance(data, dict) and "output" in data:
        texts = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text" and c.get("text"):
                        texts.append(c.get("text"))
        return "\n".join(texts).strip()
    if isinstance(data, dict) and "choices" in data:
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return ""
    return ""


def parse_gpt_indices(text: str, max_len: int) -> List[int]:
    if not text:
        return []
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            idxs = []
            for v in parsed:
                if isinstance(v, int) and 0 <= v < max_len:
                    idxs.append(v)
            return list(dict.fromkeys(idxs))
    except Exception:
        pass

    # fallback: extract numbers
    nums = [int(x) for x in re.findall(r"\d+", text)]
    idxs = [n for n in nums if 0 <= n < max_len]
    return list(dict.fromkeys(idxs))


def gpt_pick_top_indices(
    spec: SourceSpec,
    items: List[Dict[str, str]],
    top_n: int,
    extra_prompt: str,
    api_key: str,
    api_base: str,
    api_mode: str,
    model: str,
    timeout: int = 30,
) -> Tuple[List[int], Dict]:
    dbg: Dict[str, object] = {"mode": api_mode, "model": model, "top_n": top_n, "extra_prompt": extra_prompt}
    if not api_key:
        dbg["error"] = "missing_api_key"
        return [], dbg
    if not items:
        dbg["error"] = "empty_items"
        return [], dbg

    system, user = build_gpt_pick_prompt(spec, items, top_n, extra_prompt=extra_prompt)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    api_base = (api_base or "").rstrip("/")
    if api_mode == "responses":
        url = f"{api_base}/responses"
        payload = {
            "model": model,
            "input": [
                {"role": "developer", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }
    else:
        url = f"{api_base}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "developer", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }

    t0 = time.perf_counter()
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    dbg.update({"http_status": r.status_code, "elapsed_ms": elapsed_ms, "url": url})
    r.raise_for_status()

    data = r.json()
    text = extract_text_from_gpt_response(data)
    dbg["raw_text"] = text
    idxs = parse_gpt_indices(text, max_len=len(items))
    dbg["picked_indices"] = idxs
    return idxs, dbg


def format_kw_text(pairs: List[Tuple[str, int]]) -> str:
    return "\n".join([f"{pat} | {w}" for pat, w in pairs])


def parse_kw_text(text: str, fallback: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if " | " in line:
            pat, w = line.rsplit(" | ", 1)
        elif "\t" in line:
            pat, w = line.rsplit("\t", 1)
        else:
            pat, w = line, "1"
        pat = (pat or "").strip()
        if not pat:
            continue
        try:
            weight = int(str(w).strip())
        except Exception:
            weight = 1
        out.append((pat, weight))
    return out if out else fallback


def get_kw_pairs(spec: SourceSpec) -> List[Tuple[str, int]]:
    key = f"kw_{spec.key}"
    default_pairs = KW.get(spec.pick_rule, KW["generic"])
    text = st.session_state.get(key, "")
    pairs = parse_kw_text(text, default_pairs)
    st.session_state[f"kw_pairs_{spec.key}"] = pairs
    return pairs


def score_title(rule: str, title: str, kw_pairs: Optional[List[Tuple[str, int]]] = None) -> int:
    title = clean_title(title or "")
    score = 0
    pairs = kw_pairs if kw_pairs is not None else KW.get(rule, KW["generic"])
    for pat, w in pairs:
        if re.search(pat, title, flags=re.IGNORECASE):
            score += w
    return score


def format_rule_keywords_from_pairs(pairs: List[Tuple[str, int]]) -> str:
    pats = [pat for pat, _ in pairs]
    return " / ".join(pats) if pats else "-"


def auto_pick_top2(df: pd.DataFrame, rule: str, kw_pairs: Optional[List[Tuple[str, int]]] = None) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["score"] = df["title"].apply(lambda x: score_title(rule, x, kw_pairs=kw_pairs))
    top = df.sort_values(["score", "title"], ascending=[False, True]).head(2)
    df["selected"] = False
    df.loc[top.index, "selected"] = True
    return df


# -----------------------------
# UDN scraper
# -----------------------------
def find_heading_tag(soup: BeautifulSoup, heading_text: Optional[str]):
    if not heading_text:
        return None
    candidates = soup.find_all(["h1", "h2", "h3", "h4", "div", "span"])
    for tag in candidates:
        t = tag.get_text(" ", strip=True)
        if not t:
            continue
        if t == heading_text or heading_text in t:
            return tag
    return None


def extract_a_title(a) -> str:
    t = a.get_text(" ", strip=True) if a else ""
    t = re.sub(r"\s+", " ", (t or "")).strip()

    if not t:
        t = (a.get("aria-label") or a.get("title") or "").strip()

    if not t:
        img = a.find("img")
        if img:
            t = (img.get("alt") or "").strip()

    if len(t) < 6:
        h = a.find(["h2", "h3"])
        if h:
            t2 = h.get_text(" ", strip=True)
            t2 = re.sub(r"\s+", " ", (t2 or "")).strip()
            if len(t2) >= 6:
                t = t2

    return clean_title(t)


def collect_story_links_from_anchors(anchors) -> List[Dict[str, str]]:
    base = "https://udn.com"
    out = []
    for a in anchors:
        href = a.get("href", "") or ""
        if "/news/story/" not in href:
            continue
        url = norm_url(base, href)
        title = extract_a_title(a)
        title = re.sub(r"\s+", " ", (title or "")).strip()
        if title and url and len(title) >= 6:
            out.append({"title": title, "url": url})
    return dedup_keep_order(out)


def iter_story_links_between(
    soup: BeautifulSoup,
    start_heading: Optional[str],
    stop_heading: Optional[str],
    stop_candidates: Optional[List[str]],
    max_items: int = 40,
) -> List[Dict[str, str]]:
    base = "https://udn.com"
    start_tag = find_heading_tag(soup, start_heading) if start_heading else None
    stop_tag = find_heading_tag(soup, stop_heading) if stop_heading else None

    items: List[Dict[str, str]] = []
    seen = set()

    def is_story_href(href: str) -> bool:
        return bool(href) and ("/news/story/" in href)

    started = start_tag is None
    body = soup.body if soup.body else soup

    for node in body.descendants:
        if not started:
            if node == start_tag:
                started = True
            continue

        if stop_tag is not None and node == stop_tag:
            break

        if stop_candidates and getattr(node, "name", None) in ["h1", "h2", "h3", "h4"]:
            txt = node.get_text(" ", strip=True)
            if txt and start_heading and (txt != start_heading):
                if any(c in txt for c in stop_candidates):
                    break

        if getattr(node, "name", None) == "a":
            href = node.get("href", "")
            if not is_story_href(href):
                continue
            url = norm_url(base, href)
            title = extract_a_title(node)
            title = re.sub(r"\s+", " ", (title or "")).strip()
            if not title or len(title) < 6 or not url:
                continue
            if url in seen:
                continue
            seen.add(url)
            items.append({"title": title, "url": url})
            if len(items) >= max_items:
                break

    return items


def scrape_udn(spec: SourceSpec) -> Tuple[List[Dict[str, str]], Dict]:
    dbg = {"source": "udn", "url": spec.url, "start_heading": spec.start_heading, "stop_heading": spec.stop_heading}
    r, ms = http_get(spec.url, headers=DEFAULT_HEADERS, timeout=20)
    dbg.update(
        {
            "http_status": r.status_code,
            "elapsed_ms": ms,
            "final_url": r.url,
            "resp_bytes": len(r.content or b""),
        }
    )
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")

    start_tag = find_heading_tag(soup, spec.start_heading) if spec.start_heading else None
    stop_tag = find_heading_tag(soup, spec.stop_heading) if spec.stop_heading else None

    dbg.update(
        {
            "start_heading_found": bool(start_tag),
            "stop_heading_found": bool(stop_tag),
            "stop_heading_candidates": spec.stop_heading_candidates,
            "page_story_link_count": len(soup.select('a[href*="/news/story/"]')),
        }
    )

    items: List[Dict[str, str]] = []

    # ✅ FIX: 若是「最新文章」(start_heading=None, stop_heading 有值)，用 stop_tag 的 previous links 抓，避免 DOM 先遇到 stop_tag 直接 break
    if spec.start_heading is None and spec.stop_heading and stop_tag is not None:
        prev_anchors = stop_tag.find_all_previous("a", href=re.compile(r"/news/story/"))
        # find_all_previous 是倒序（由近到遠），要反轉成頁面順序
        prev_anchors = list(reversed(prev_anchors))
        items = collect_story_links_from_anchors(prev_anchors)
        dbg["udn_latest_strategy"] = "stop_tag.find_all_previous"
        dbg["items_raw_count"] = len(items)
        dbg["items_dedup_count"] = len(items)
        dbg["sample_items"] = items[:10]
        # 若 still empty，fallback 全頁
        if not items:
            all_anchors = soup.select('a[href*="/news/story/"]')
            items = collect_story_links_from_anchors(all_anchors)
            dbg["udn_latest_fallback"] = "all_story_links"
            dbg["items_raw_count"] = len(items)
            dbg["items_dedup_count"] = len(items)
            dbg["sample_items"] = items[:10]
        return items[:30], dbg

    # 其他區塊：原本 between-heading 方法
    items_raw = iter_story_links_between(
        soup,
        start_heading=spec.start_heading,
        stop_heading=spec.stop_heading,
        stop_candidates=spec.stop_heading_candidates,
        max_items=80,
    )
    dbg["items_raw_count"] = len(items_raw)

    items = dedup_keep_order(items_raw)
    dbg["items_dedup_count"] = len(items)
    dbg["sample_items"] = items[:10]

    # fallback：若 between-heading 抓不到但全頁有 story links，至少給全頁前 N
    if not items and dbg.get("page_story_link_count", 0) > 0:
        all_anchors = soup.select('a[href*="/news/story/"]')
        items = collect_story_links_from_anchors(all_anchors)
        dbg["udn_fallback"] = "all_story_links"
        dbg["items_raw_count"] = len(items)
        dbg["items_dedup_count"] = len(items)
        dbg["sample_items"] = items[:10]

    return items[:30], dbg


# -----------------------------
# CNYES category API
# -----------------------------
def extract_list_candidates(obj):
    found = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            if x and all(isinstance(i, dict) for i in x):
                keys = set().union(*[set(i.keys()) for i in x])
                if "title" in keys and ("link" in keys or "newsId" in keys or "id" in keys or "url" in keys):
                    found.append(x)
            for i in x:
                walk(i)

    walk(obj)
    return found


def parse_cnyes_items(data: dict) -> List[Dict[str, str]]:
    cand = None
    if isinstance(data, dict):
        if isinstance(data.get("items"), dict) and isinstance(data["items"].get("data"), list):
            cand = data["items"]["data"]
        elif isinstance(data.get("data"), dict) and isinstance(data["data"].get("items"), list):
            cand = data["data"]["items"]

    if cand is None:
        lists = extract_list_candidates(data)
        cand = lists[0] if lists else []

    items = []
    for it in cand or []:
        title = clean_title(it.get("title") or "")
        link = (it.get("link") or it.get("url") or "").strip()
        news_id = it.get("newsId") or it.get("id")
        if not link and news_id and str(news_id).isdigit():
            link = f"https://news.cnyes.com/news/id/{news_id}"
        if title and link:
            items.append({"title": title, "url": link})
    return dedup_keep_order(items)


def scrape_cnyes_category(spec: SourceSpec, secret: str = "") -> Tuple[List[Dict[str, str]], Dict]:
    slug = spec.cnyes_slug or "news24h"
    api = f"https://api.cnyes.com/media/api/v1/newslist/category/{slug}"
    params = {"page": "1", "limit": "30"}
    if secret:
        params["secret"] = secret

    headers = {**DEFAULT_HEADERS, "Origin": "https://news.cnyes.com", "Referer": "https://news.cnyes.com/"}
    dbg = {"source": "cnyes_api", "slug": slug, "api": api, "params": params}

    r, ms = http_get(api, headers=headers, params=params, timeout=20)
    dbg.update(
        {
            "http_status": r.status_code,
            "elapsed_ms": ms,
            "final_url": r.url,
            "resp_bytes": len(r.content or b""),
        }
    )
    r.raise_for_status()

    data = r.json()
    items = parse_cnyes_items(data)
    dbg.update({"items_count": len(items), "sample_items": items[:10]})
    return items, dbg


# -----------------------------
# CNYES search page (first news) - 多重 fallback
# -----------------------------
def deep_find_news_items(obj):
    found = []

    def walk(x):
        if isinstance(x, dict):
            title = x.get("title")
            url = x.get("url") or x.get("link")
            news_id = x.get("newsId") or x.get("id")

            if isinstance(title, str) and title.strip():
                title = clean_title(title)
                if isinstance(url, str) and url.strip():
                    found.append({"title": title.strip(), "url": url.strip()})
                elif news_id and str(news_id).isdigit():
                    found.append({"title": title.strip(), "url": f"https://news.cnyes.com/news/id/{news_id}"})

            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for i in x:
                walk(i)

    walk(obj)
    return dedup_keep_order(found)


def scrape_cnyes_search_first_news(search_url: str, keyword: str = "") -> Tuple[List[Dict[str, str]], Dict]:
    dbg = {"source": "cnyes_search", "url": search_url, "keyword": keyword}

    headers = {**DEFAULT_HEADERS, "Referer": "https://www.cnyes.com/", "Origin": "https://www.cnyes.com"}
    r, ms = http_get(search_url, headers=headers, timeout=20)
    dbg.update(
        {
            "http_status": r.status_code,
            "elapsed_ms": ms,
            "final_url": r.url,
            "resp_bytes": len(r.content or b""),
        }
    )
    r.raise_for_status()

    html = r.text or ""
    soup = BeautifulSoup(html, "lxml")

    # Method 1: HTML anchors
    items = []
    for a in soup.select('a[href*="/news/id/"], a[href*="news/id/"]'):
        href = a.get("href", "") or ""
        title = a.get_text(" ", strip=True)
        title = re.sub(r"\s+", " ", (title or "")).strip()
        if not title:
            title = (a.get("aria-label") or a.get("title") or "").strip()
            if not title:
                img = a.find("img")
                if img:
                    title = (img.get("alt") or "").strip()
        title = clean_title(title)
        if not title:
            continue

        if href.startswith("http"):
            url = href
        else:
            url = norm_url("https://news.cnyes.com", href)

        items.append({"title": title, "url": url})

    items = dedup_keep_order(items)
    dbg["method"] = "html_anchor"
    dbg["html_anchor_candidates"] = len(items)
    dbg["sample_candidates"] = items[:10]

    if items:
        if keyword:
            for it in items:
                if keyword in it["title"]:
                    dbg["picked_reason"] = "first_match_keyword_in_html_anchor"
                    return [it], dbg
            dbg["picked_reason"] = "no_keyword_match_use_first_html_anchor"
            return [items[0]], dbg
        dbg["picked_reason"] = "use_first_html_anchor"
        return [items[0]], dbg

    # Method 2: Regex find any news url inside HTML
    url_candidates = re.findall(r"https?://news\.cnyes\.com/news/id/\d+", html)
    url_candidates = list(dict.fromkeys(url_candidates))
    dbg["regex_url_candidates"] = len(url_candidates)
    dbg["regex_url_sample"] = url_candidates[:10]

    if url_candidates:
        first_url = url_candidates[0]
        dbg["picked_reason"] = "regex_url_found_use_first_url"

        idx = html.find(first_url)
        title = ""
        if idx != -1:
            window = html[max(0, idx - 2000) : idx + 2000]
            m = re.search(r'"title"\s*:\s*"([^"]+)"', window)
            if m:
                title = m.group(1)
                title = clean_title(title)

        if not title:
            title = keyword or "CNYES News"

        return [{"title": title, "url": first_url}], dbg

    # Method 3: __NEXT_DATA__ fallback
    script = soup.find("script", id="__NEXT_DATA__")
    if script:
        raw = script.get_text(strip=True)
        dbg["next_data_bytes"] = len(raw.encode("utf-8", errors="ignore")) if raw else 0
        if raw:
            try:
                data = json.loads(raw)
                candidates = deep_find_news_items(data)
                dbg["method"] = "__NEXT_DATA__"
                dbg["next_data_candidates"] = len(candidates)
                dbg["sample_candidates"] = candidates[:10]

                if candidates:
                    if keyword:
                        for it in candidates:
                            if keyword in it["title"]:
                                dbg["picked_reason"] = "first_match_keyword_in_next_data"
                                return [it], dbg
                        dbg["picked_reason"] = "no_keyword_match_use_first_next_data"
                        return [candidates[0]], dbg
                    dbg["picked_reason"] = "use_first_next_data"
                    return [candidates[0]], dbg
            except Exception as e:
                dbg["method"] = "__NEXT_DATA__"
                dbg["next_data_parse_error"] = f"{type(e).__name__}: {e}"

    # Method 4: Loose regex for (newsId,title) pairs
    dbg["method"] = "regex_newsId_title"
    pairs = []
    for m in re.finditer(r'"newsId"\s*:\s*(\d+)', html):
        news_id = m.group(1)
        start = max(0, m.start() - 1200)
        end = min(len(html), m.end() + 1200)
        chunk = html[start:end]
        tm = re.search(r'"title"\s*:\s*"([^"]+)"', chunk)
        if tm:
            title = tm.group(1)
            title = clean_title(title)
            url = f"https://news.cnyes.com/news/id/{news_id}"
            pairs.append({"title": title, "url": url})
        if len(pairs) >= 30:
            break

    pairs = dedup_keep_order(pairs)
    dbg["regex_pairs_count"] = len(pairs)
    dbg["sample_candidates"] = pairs[:10]

    if pairs:
        if keyword:
            for it in pairs:
                if keyword in it["title"]:
                    dbg["picked_reason"] = "first_match_keyword_in_regex_pairs"
                    return [it], dbg
            dbg["picked_reason"] = "no_keyword_match_use_first_regex_pairs"
            return [pairs[0]], dbg
        dbg["picked_reason"] = "use_first_regex_pairs"
        return [pairs[0]], dbg

    dbg["picked_reason"] = "no_candidates_found"
    return [], dbg


# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="新聞爬取與挑選（title+url 匯出）", layout="wide")
st.title("新聞爬取與挑選（可局部修改＋匯出 title/url）")


def init_state():
    for s in SOURCES_UI:
        if s.key not in st.session_state:
            st.session_state[s.key] = pd.DataFrame(columns=["selected", "title", "url", "score"])
        dbg_key = f"debug_{s.key}"
        if dbg_key not in st.session_state:
            st.session_state[dbg_key] = {}
        kw_key = f"kw_{s.key}"
        if kw_key not in st.session_state:
            st.session_state[kw_key] = format_kw_text(KW.get(s.pick_rule, KW["generic"]))


init_state()

with st.sidebar:
    st.subheader("控制面板")
    secret = st.text_input("（選用）鉅亨 OPEN API secret", value="", type="password")
    export_fmt = st.selectbox("匯出格式", ["CSV", "XLSX", "JSON"], index=0)
    enable_autopick = st.toggle("抓取後自動挑選 Top 2（每區塊）", value=True)
    use_gpt_pick = st.toggle("使用 GPT 判斷 Top2", value=False)
    gpt_api_key = st.text_input("GPT API Key", value=DEFAULT_GPT_KEY, type="password")
    gpt_api_base = st.text_input("GPT API Base URL", value=DEFAULT_GPT_BASE)
    gpt_api_mode = st.selectbox("GPT API 介面", ["responses", "chat_completions"], index=0)
    gpt_model = st.text_input("GPT Model", value=DEFAULT_GPT_MODEL)
    max_keep = st.slider("每區塊保留候選條數", 10, 60, 30, step=5)
    show_debug = st.toggle("顯示 Debug 面板", value=True)
    if use_gpt_pick and not gpt_api_key:
        st.warning("已啟用 GPT 判斷，但未填 API Key。將改用規則自動挑選。")
    st.divider()
    run_all = st.button("一鍵：全部抓取", type="primary")


def set_df_for_source(spec: SourceSpec, items: List[Dict[str, str]]):
    df = pd.DataFrame(items)
    if df.empty:
        df = pd.DataFrame(columns=["title", "url"])

    df = df.head(max_keep)

    if "selected" not in df.columns:
        df.insert(0, "selected", False)
    else:
        df["selected"] = df["selected"].fillna(False)

    kw_pairs = get_kw_pairs(spec)
    if not df.empty:
        df["score"] = df["title"].apply(lambda x: score_title(spec.pick_rule, x, kw_pairs=kw_pairs))
    else:
        df["score"] = pd.Series(dtype="int")

    gpt_dbg = {}
    if enable_autopick and not df.empty:
        top_n = 2 if len(df) >= 2 else len(df)
        if use_gpt_pick and gpt_api_key:
            try:
                extra_prompt = st.session_state.get(f"prompt_{spec.key}", "")
                idxs, gpt_dbg = gpt_pick_top_indices(
                    spec=spec,
                    items=df[["title", "url"]].to_dict(orient="records"),
                    top_n=top_n,
                    extra_prompt=extra_prompt,
                    api_key=gpt_api_key,
                    api_base=gpt_api_base,
                    api_mode=gpt_api_mode,
                    model=gpt_model,
                )
                if idxs:
                    df["selected"] = False
                    df.loc[idxs, "selected"] = True
                else:
                    df = auto_pick_top2(df, spec.pick_rule, kw_pairs=kw_pairs)
            except Exception as e:
                gpt_dbg = {"error": f"{type(e).__name__}: {e}"}
                df = auto_pick_top2(df, spec.pick_rule, kw_pairs=kw_pairs)
        else:
            df = auto_pick_top2(df, spec.pick_rule, kw_pairs=kw_pairs)

    st.session_state[spec.key] = df[["selected", "title", "url", "score"]].copy()
    if use_gpt_pick:
        st.session_state[f"gpt_{spec.key}"] = gpt_dbg
    else:
        st.session_state.pop(f"gpt_{spec.key}", None)


def apply_pick(df: pd.DataFrame, spec: SourceSpec) -> Tuple[pd.DataFrame, Dict]:
    if df is None or df.empty:
        return df, {}
    top_n = 2 if len(df) >= 2 else len(df)
    kw_pairs = get_kw_pairs(spec)
    if use_gpt_pick and gpt_api_key:
        try:
            extra_prompt = st.session_state.get(f"prompt_{spec.key}", "")
            idxs, gpt_dbg = gpt_pick_top_indices(
                spec=spec,
                items=df[["title", "url"]].to_dict(orient="records"),
                top_n=top_n,
                extra_prompt=extra_prompt,
                api_key=gpt_api_key,
                api_base=gpt_api_base,
                api_mode=gpt_api_mode,
                model=gpt_model,
            )
            if idxs:
                df = df.copy()
                df["selected"] = False
                df.loc[idxs, "selected"] = True
                return df, gpt_dbg
            return auto_pick_top2(df, spec.pick_rule, kw_pairs=kw_pairs), gpt_dbg
        except Exception as e:
            gpt_dbg = {"error": f"{type(e).__name__}: {e}"}
            return auto_pick_top2(df, spec.pick_rule, kw_pairs=kw_pairs), gpt_dbg
    return auto_pick_top2(df, spec.pick_rule, kw_pairs=kw_pairs), {}


def scrape_one(spec: SourceSpec) -> Tuple[bool, str]:
    try:
        if spec.mode == "udn_html":
            items, dbg = scrape_udn(spec)
        elif spec.mode == "cnyes_api":
            items, dbg = scrape_cnyes_category(spec, secret=secret)
        elif spec.mode == "cnyes_keyword":
            items, dbg = scrape_cnyes_search_first_news(spec.url, keyword=spec.keyword or "")
        else:
            items, dbg = [], {"source": "unknown", "url": spec.url}

        st.session_state[f"debug_{spec.key}"] = dbg
        set_df_for_source(spec, items)

        if not items:
            return False, "抓取成功但結果為空（請看 Debug 面板）"
        return True, f"OK ({len(items)} items)"
    except Exception as e:
        st.session_state[f"debug_{spec.key}"] = {"error": f"{type(e).__name__}: {e}", "url": spec.url}
        set_df_for_source(spec, [])
        return False, f"{type(e).__name__}: {e}"


if run_all:
    for spec in SOURCES_UI:
        ok, msg = scrape_one(spec)
        if not ok:
            st.warning(f"{msg}")


def render_source_block(spec: SourceSpec):
    st.markdown(f"### {spec.name}")
    pick_method = "GPT 判斷" if use_gpt_pick and gpt_api_key else "規則自動挑選"
    st.caption(f"挑選方法：{pick_method}")
    kw_key = f"kw_{spec.key}"
    if kw_key not in st.session_state:
        st.session_state[kw_key] = format_kw_text(KW.get(spec.pick_rule, KW["generic"]))
    st.text_area(
        "關鍵字（可編輯，每行：正則式 | 權重）",
        key=kw_key,
        height=120,
        placeholder="例如：台積電|TSMC|2330 | 8",
    )
    kw_pairs = get_kw_pairs(spec)
    st.caption(f"關鍵字（規則）：{format_rule_keywords_from_pairs(kw_pairs)}")
    prompt_key = f"prompt_{spec.key}"
    if prompt_key not in st.session_state:
        st.session_state[prompt_key] = ""
    st.text_area(
        "GPT 額外挑選指令（可留空）",
        key=prompt_key,
        height=80,
        help="修改後請按「重新套用 Top2」以重新挑選",
        placeholder="例如：優先選擇含『盤後』『收盤』且有數字的標題",
    )
    if st.session_state.get(prompt_key, "").strip() and not (use_gpt_pick and gpt_api_key):
        st.caption("注意：尚未啟用 GPT 判斷，此指令不會生效。")
    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        if st.button("抓取（只跑這一塊）", key=f"btn_fetch_{spec.key}"):
            ok, msg = scrape_one(spec)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    with col2:
        if st.button("重新套用 Top2", key=f"btn_pick_{spec.key}"):
            df = st.session_state[spec.key].copy()
            if not df.empty:
                df, gpt_dbg = apply_pick(df, spec)
                st.session_state[spec.key] = df
                if gpt_dbg:
                    st.session_state[f"gpt_{spec.key}"] = gpt_dbg
                else:
                    st.session_state.pop(f"gpt_{spec.key}", None)
                st.toast("已重新挑選 Top2", icon="✅")

    df = st.session_state[spec.key].copy()

    if df.empty:
        st.info("尚未抓取，或抓取結果為空。")
        if show_debug:
            dbg = st.session_state.get(f"debug_{spec.key}", {})
            with st.expander("Debug 面板", expanded=False):
                st.json(dbg if dbg else {"info": "尚無 Debug 資訊"})
        st.divider()
        return

    edited = st.data_editor(
        df,
        key=f"editor_{spec.key}",
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "selected": st.column_config.CheckboxColumn("selected"),
            "title": st.column_config.TextColumn("title"),
            "url": st.column_config.TextColumn("url"),
            "score": st.column_config.NumberColumn("score", help="自動打分用（可忽略）"),
        },
    )
    st.session_state[spec.key] = edited

    picked = edited[edited["selected"] == True][["title", "url"]].copy()
    st.caption(f"已選：{len(picked)} 則")

    if show_debug:
        dbg = st.session_state.get(f"debug_{spec.key}", {})
        gpt_dbg = st.session_state.get(f"gpt_{spec.key}", {})
        with st.expander("Debug 面板", expanded=False):
            if not dbg:
                st.info("尚無 Debug 資訊（尚未抓取或未寫入）。")
            else:
                cols = st.columns(4)
                cols[0].metric("HTTP", dbg.get("http_status", "-"))
                cols[1].metric("耗時(ms)", dbg.get("elapsed_ms", "-"))
                cols[2].metric("回應大小(bytes)", dbg.get("resp_bytes", "-"))
                items_cnt = dbg.get(
                    "items_dedup_count",
                    dbg.get("items_count", dbg.get("html_anchor_candidates", dbg.get("next_data_candidates", "-"))),
                )
                cols[3].metric("候選數", items_cnt)
                st.json(dbg)
            if gpt_dbg:
                st.markdown("GPT Debug")
                st.json(gpt_dbg)

    st.divider()


for spec in SOURCES_UI:
    render_source_block(spec)


# -----------------------------
# Final export
# -----------------------------
st.subheader("最終匯出清單（彙總所有 selected）")

final_rows = []
for spec in SOURCES_UI:
    df = st.session_state[spec.key]
    if df is None or df.empty:
        continue
    picked = df[df["selected"] == True][["title", "url"]].copy()
    for _, r in picked.iterrows():
        t = clean_title(r["title"] or "")
        u = (r["url"] or "").strip()
        if t and u:
            final_rows.append({"title": t, "url": u})

final_df = pd.DataFrame(final_rows)
if final_df.empty:
    st.info("目前尚無 selected 項目。請先抓取並勾選。")
else:
    final_df = final_df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    final_df = st.data_editor(
        final_df,
        key="final_editor",
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "title": st.column_config.TextColumn("title"),
            "url": st.column_config.TextColumn("url"),
        },
    )

    def to_csv_bytes(df_: pd.DataFrame) -> bytes:
        return df_.to_csv(index=False).encode("utf-8-sig")

    def to_xlsx_bytes(df_: pd.DataFrame) -> bytes:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_.to_excel(writer, index=False, sheet_name="export")
        return output.getvalue()

    def to_json_bytes(df_: pd.DataFrame) -> bytes:
        return json.dumps(df_.to_dict(orient="records"), ensure_ascii=False, indent=2).encode("utf-8")

    if export_fmt == "CSV":
        data = to_csv_bytes(final_df)
        mime = "text/csv"
        fname = "news_export.csv"
    elif export_fmt == "XLSX":
        data = to_xlsx_bytes(final_df)
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        fname = "news_export.xlsx"
    else:
        data = to_json_bytes(final_df)
        mime = "application/json"
        fname = "news_export.json"

    st.download_button(
        label=f"下載匯出檔（{export_fmt}）",
        data=data,
        file_name=fname,
        mime=mime,
        type="primary",
    )

st.caption(
    "提醒：UDN 分區 DOM 可能讓 stop heading 先出現，最新版已對「最新文章」改用 stop_tag 的 previous links 取值，避免 items=0。鉅亨搜尋頁仍可能前端渲染，本版已加多重 fallback。"
)
