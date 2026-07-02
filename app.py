import io
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

USER_AGENT = "HPAnalyzerMVP/0.3 (+https://example.com)"
TIMEOUT = 12

KEYWORDS = {
    "price": ["料金", "費用", "報酬", "価格", "プラン", "見積", "相談料"],
    "case": ["事例", "実績", "解決", "お客様の声", "口コミ", "レビュー", "相談実績"],
    "flow": ["流れ", "手順", "ご相談の流れ", "依頼の流れ", "進め方"],
    "faq": ["よくある質問", "FAQ", "Q&A", "質問"],
    "profile": ["代表", "専門家", "弁護士", "司法書士", "税理士", "行政書士", "スタッフ", "プロフィール", "事務所紹介"],
    "cta": ["無料相談", "お問い合わせ", "問合せ", "相談する", "電話", "LINE", "フォーム", "予約", "メール"],
    "trust": ["資格", "登録番号", "所属", "認定", "受賞", "掲載", "監修", "執筆", "対応件数", "年間"],
    "area": ["市", "区", "町", "村", "県", "対応エリア", "地域"],
}

BAD_GENERIC_PHRASES = [
    "親切丁寧", "地域密着", "お気軽にご相談", "迅速丁寧", "安心サポート", "お客様第一"
]

@dataclass
class PageData:
    url: str
    title: str
    description: str
    h1: List[str]
    h2: List[str]
    text: str
    links: List[Tuple[str, str]]
    html: str


def normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def fetch_page(url: str) -> PageData:
    headers = {"User-Agent": USER_AGENT}
    res = requests.get(url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "lxml")

    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    meta = soup.find("meta", attrs={"name": re.compile("description", re.I)})
    description = meta.get("content", "").strip() if meta else ""
    h1 = [x.get_text(" ", strip=True) for x in soup.find_all("h1")]
    h2 = [x.get_text(" ", strip=True) for x in soup.find_all("h2")]
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

    links = []
    for a in soup.find_all("a", href=True):
        label = a.get_text(" ", strip=True)
        href = urljoin(res.url, a["href"])
        links.append((label, href))

    return PageData(url=res.url, title=title, description=description, h1=h1, h2=h2, text=text, links=links, html=res.text)


def same_domain(base: str, target: str) -> bool:
    return urlparse(base).netloc.replace("www.", "") == urlparse(target).netloc.replace("www.", "")


def select_important_links(home: PageData, max_pages: int) -> List[str]:
    candidates = []
    wanted = KEYWORDS["price"] + KEYWORDS["case"] + KEYWORDS["flow"] + KEYWORDS["faq"] + KEYWORDS["profile"] + ["相続", "離婚", "刑事", "交通事故", "債務", "登記", "生前対策", "業務", "サービス"]
    for label, href in home.links:
        if not same_domain(home.url, href):
            continue
        if any(x in (label + href) for x in wanted):
            clean = href.split("#")[0]
            if clean and clean != home.url:
                candidates.append(clean)
    seen = []
    for c in candidates:
        if c not in seen:
            seen.append(c)
    return seen[:max_pages]


def count_hits(text: str, words: List[str]) -> int:
    return sum(1 for w in words if w.lower() in text.lower())


def has_phone(text: str, html: str) -> bool:
    phone_pattern = r"(0\d{1,4}[-ー−]?\d{1,4}[-ー−]?\d{3,4})"
    return bool(re.search(phone_pattern, text)) or "tel:" in html.lower()


def has_contact_link_or_mail(html: str, text: str) -> bool:
    return "mailto:" in html.lower() or any(k in text for k in ["お問い合わせ", "問合せ", "メールフォーム", "フォーム"])


def has_real_form(html: str) -> bool:
    # 「お問い合わせ」という文字だけではフォームありと判定しない。実際のformタグを重視。
    return "<form" in html.lower()


def has_line(html: str, text: str) -> bool:
    return "line.me" in html.lower() or "lin.ee" in html.lower() or "LINE" in text


def page_flags(pages: List[PageData]) -> Dict[str, bool]:
    all_text = " ".join(p.text for p in pages)
    all_html = " ".join(p.html for p in pages)
    all_links = " ".join(" ".join([label, href]) for p in pages for label, href in p.links)
    all_headings = " ".join(" ".join(p.h1 + p.h2) for p in pages)
    all_urls = " ".join(p.url for p in pages) + " " + " ".join(href for p in pages for _, href in p.links)
    joined = all_text + " " + all_links + " " + all_headings + " " + all_urls

    # 電話番号などの数字で「料金あり」と誤判定しないよう、円・万円・税込などの近接表現のみを見る。
    money_pattern = r"([0-9０-９][0-9０-９,，]*(?:\s)*(?:円|万円)|¥\s*[0-9０-９,，]+|￥\s*[0-9０-９,，]+|税込|税別)"
    has_money = bool(re.search(money_pattern, joined))
    form_tag = has_real_form(all_html)
    img_count = all_html.lower().count("<img")
    table_count = all_html.lower().count("<table")
    text_len = len(all_text)

    # FAQ・相談の流れは「存在確認」と「十分な説明量」を分ける。
    # お客様向けレポートで、既にある項目を「追加する」と誤表示しないため。
    flow_terms = KEYWORDS["flow"] + ["初回相談", "ご依頼", "面談", "ヒアリング", "必要書類", "完了まで", "申告まで"]
    faq_terms = KEYWORDS["faq"] + ["よくいただく質問", "よく頂く質問", "ご質問", "質問と回答"]
    flow_exists = count_hits(joined, flow_terms) >= 1 or bool(re.search(r"/(flow|step|guide|nagare|procedure|process)(/|$|-|_)", all_urls, re.I))
    faq_exists = count_hits(joined, faq_terms) >= 1 or bool(re.search(r"/(faq|qa|q-a|question)(/|$|-|_)", all_urls, re.I))
    flow_detail = count_hits(joined, flow_terms) >= 3
    faq_detail = count_hits(joined, faq_terms) >= 3

    return {
        # 単語の有無ではなく、営業資料で根拠にできる水準まで要求する。
        "has_price": count_hits(joined, KEYWORDS["price"]) >= 2 and has_money,
        "has_case": count_hits(joined, KEYWORDS["case"]) >= 4,
        "has_flow": flow_exists,
        "has_flow_detail": flow_detail,
        "has_faq": faq_exists,
        "has_faq_detail": faq_detail,
        "has_profile": count_hits(joined, KEYWORDS["profile"]) >= 4,
        "has_trust": count_hits(joined, KEYWORDS["trust"]) >= 4,
        "has_phone": has_phone(all_text, all_html),
        "has_contact_link": has_contact_link_or_mail(all_html, all_text),
        "has_form": form_tag,
        "has_line": has_line(all_html, all_text),
        "has_ssl": pages[0].url.startswith("https://"),
        "is_image_heavy": img_count >= 45 and text_len < img_count * 180,
        "is_table_old_layout": table_count >= 8,
        "has_money_expression": has_money,
    }


def cap(value: int, upper: int) -> int:
    return max(0, min(value, upper))


def exact_or_partial_hit(needle: str, haystack: str) -> bool:
    if not needle:
        return False
    needle = needle.strip()
    if not needle:
        return False
    # 入力された業務・地域を重視。完全一致のほか、スペース区切り語の一部一致も見る。
    if needle in haystack:
        return True
    parts = [x for x in re.split(r"[\s　・,/、]+", needle) if len(x) >= 2]
    return any(x in haystack for x in parts)



def apply_strictness_to_scores(scores: Dict[str, int], strictness: str) -> Dict[str, int]:
    """URLだけの簡易診断で点数が甘く出すぎる問題を抑えるための補正。"""
    if strictness in ["簡易診断", "標準"]:
        return scores
    if strictness in ["通常診断", "辛口"]:
        return {k: cap(round(v * 0.88), 88) for k, v in scores.items()}
    # 詳細診断：単語検出だけで高得点にしない。かなり厳しめ。
    return {k: cap(round(v * 0.76), 82) for k, v in scores.items()}


def strictness_total_cap(strictness: str) -> int:
    if strictness in ["簡易診断", "標準"]:
        return 100
    if strictness in ["通常診断", "辛口"]:
        return 84
    # URLだけでは実CVR・流入・広告成果・スマホ実表示までは見られないため、詳細診断は上限を設ける。
    return 76

def score_site(pages: List[PageData], industry: str, area: str, service: str, strictness: str = "詳細診断") -> Dict:
    home = pages[0]
    flags = page_flags(pages)
    all_text = " ".join(p.text for p in pages)
    fv_text = " ".join([home.title, home.description] + home.h1[:2] + home.h2[:4])

    # 以前は「市」「区」などがあるだけで地域評価が入り、採点が甘くなっていたため、
    # 入力された対応エリアの一致を最優先にする。
    area_hit = exact_or_partial_hit(area, all_text) if area else False
    service_hit = exact_or_partial_hit(service, all_text) if service else False
    industry_hit = bool(industry and industry != "その他" and industry in all_text)
    cta_count = count_hits(all_text, KEYWORDS["cta"])
    fv_cta_count = count_hits(fv_text, KEYWORDS["cta"])
    generic_penalty = min(15, count_hits(fv_text, BAD_GENERIC_PHRASES) * 5)

    scores = {}

    fv = (
        18 * bool(home.h1)
        + 28 * service_hit
        + 22 * area_hit
        + 18 * (fv_cta_count >= 1)
        + 8 * ("無料相談" in fv_text or "初回相談" in fv_text)
        + 6 * (len(fv_text) >= 50)
        - generic_penalty
    )
    if not service_hit:
        fv = min(fv, 58)
    if not area_hit:
        fv = min(fv, 68)
    if fv_cta_count == 0:
        fv = min(fv, 72)
    scores["ファーストビュー"] = cap(fv, 100)

    appeal = (
        25 * service_hit
        + 18 * area_hit
        + 15 * (count_hits(all_text, ["悩み", "不安", "お困り", "期限", "手続き", "解決"]) >= 3)
        + 15 * (count_hits(all_text, ["強み", "選ばれる", "理由", "特徴"]) >= 2)
        + 12 * ("無料相談" in all_text or "初回相談" in all_text)
        + 10 * (cta_count >= 5)
        + 5 * industry_hit
    )
    if not service_hit:
        appeal = min(appeal, 55)
    if not area_hit:
        appeal = min(appeal, 70)
    scores["訴求力"] = cap(appeal, 100)

    trust = (
        18 * flags["has_profile"]
        + 20 * flags["has_case"]
        + 18 * flags["has_trust"]
        + 12 * ("写真" in all_text or "プロフィール" in all_text or "代表" in all_text)
        + 8 * flags["has_faq"]
        + 4 * flags.get("has_faq_detail", False)
        + 8 * flags["has_flow"]
        + 4 * flags.get("has_flow_detail", False)
        + 8 * ("料金" in all_text or "費用" in all_text)
    )
    if not flags["has_profile"]:
        trust = min(trust, 62)
    if not flags["has_case"]:
        trust = min(trust, 70)
    if not flags["has_trust"]:
        trust = min(trust, 78)
    scores["信頼力"] = cap(trust, 100)

    inquiry = (
        22 * flags["has_phone"]
        + 18 * flags["has_contact_link"]
        + 18 * flags["has_form"]
        + 8 * flags["has_line"]
        + 14 * (cta_count >= 5)
        + 10 * ("無料相談" in all_text or "初回相談" in all_text)
        + 10 * ("予約" in all_text or "相談する" in all_text)
    )
    if not (flags["has_phone"] and flags["has_contact_link"]):
        inquiry = min(inquiry, 62)
    if not flags["has_form"]:
        inquiry = min(inquiry, 74)
    if cta_count < 3:
        inquiry = min(inquiry, 68)
    scores["問い合わせ力"] = cap(inquiry, 100)

    price = (
        46 * flags["has_price"]
        + 12 * ("相談料" in all_text or "無料相談" in all_text)
        + 12 * ("見積" in all_text)
        + 10 * ("追加" in all_text or "実費" in all_text)
        + 10 * ("税込" in all_text or "税別" in all_text)
        + 10 * flags["has_money_expression"]
    )
    if not flags["has_price"]:
        price = min(price, 32)
    scores["料金の分かりやすさ"] = cap(price, 100)

    content = (
        12 * (len(pages) >= 3)
        + 14 * (len(pages) >= 5)
        + 9 * flags["has_faq"]
        + 5 * flags.get("has_faq_detail", False)
        + 9 * flags["has_flow"]
        + 5 * flags.get("has_flow_detail", False)
        + 16 * flags["has_case"]
        + 15 * (len(all_text) >= 5000)
        + 15 * (len(home.h2) >= 5)
    )
    if len(all_text) < 2500:
        content = min(content, 55)
    scores["コンテンツの充実度"] = cap(content, 100)

    seo = (
        16 * bool(home.title)
        + 12 * bool(home.description)
        + 18 * bool(home.h1)
        + 12 * (len(home.h1) == 1)
        + 12 * (len(home.h2) >= 3)
        + 18 * (area_hit and service_hit)
        + 12 * (service_hit and (service in home.title or any(service in h for h in home.h1))) if service else 0
    )
    if not (area_hit and service_hit):
        seo = min(seo, 68)
    scores["SEO内部構造"] = cap(seo, 100)

    tech = (
        25 * flags["has_ssl"]
        + 17 * (len(home.title) <= 45 and len(home.title) >= 10)
        + 17 * (len(home.description) <= 140 and len(home.description) >= 50)
        + 16 * ("viewport" in home.html.lower())
        + 15 * (home.html.lower().count("alt=") >= 5)
        + 10 * ("canonical" in home.html.lower())
    )
    scores["技術・基本品質"] = cap(tech, 100)

    # 画像バナー依存・古いtableレイアウトが強い場合は、URLテキスト上では情報があっても詳細診断では厳しく見る。
    if flags.get("is_image_heavy"):
        scores["ファーストビュー"] = min(scores["ファーストビュー"], 72)
        scores["問い合わせ力"] = min(scores["問い合わせ力"], 72)
        scores["技術・基本品質"] = min(scores["技術・基本品質"], 62)
    if flags.get("is_table_old_layout"):
        scores["技術・基本品質"] = min(scores["技術・基本品質"], 58)

    scores = apply_strictness_to_scores(scores, strictness)

    weights = {
        "ファーストビュー": 0.16,
        "訴求力": 0.15,
        "信頼力": 0.17,
        "問い合わせ力": 0.18,
        "料金の分かりやすさ": 0.12,
        "コンテンツの充実度": 0.09,
        "SEO内部構造": 0.08,
        "技術・基本品質": 0.05,
    }
    total = sum(scores[k] * weights[k] for k in scores)

    # 重要項目の欠落は総合点にも反映。重要項目が不足しているのに「何となく60点台」にならないようにする。
    penalties = []
    if not service_hit:
        penalties.append(("狙いたい業務の訴求が弱い", 8))
    if not area_hit:
        penalties.append(("対応エリアの訴求が弱い", 6))
    if not flags["has_price"]:
        penalties.append(("料金情報が弱い", 10))
    if not flags["has_case"]:
        penalties.append(("事例・実績が弱い", 8))
    if not flags["has_profile"]:
        penalties.append(("専門家プロフィールが弱い", 5))
    if not (flags["has_phone"] and flags["has_contact_link"]):
        penalties.append(("問い合わせ導線が不足", 9))
    if not flags["has_form"]:
        penalties.append(("実フォームの確認が弱い", 5))
    if flags.get("is_image_heavy"):
        penalties.append(("画像バナー依存が強く、テキスト評価だけでは過大評価されやすい", 6))
    if flags.get("is_table_old_layout"):
        penalties.append(("古いHTMLレイアウトの可能性", 4))
    if not flags["has_flow"]:
        penalties.append(("相談の流れが未確認", 4))
    elif not flags.get("has_flow_detail", False):
        penalties.append(("相談の流れの説明量が不足", 2))
    if cta_count < 3:
        penalties.append(("CTAが少ない", 4))

    total = round(max(0, total - sum(p[1] for p in penalties)))
    total = min(total, strictness_total_cap(strictness))
    rank = "A" if total >= 90 else "B" if total >= 75 else "C" if total >= 60 else "D" if total >= 40 else "E"
    return {
        "scores": scores,
        "total": total,
        "rank": rank,
        "flags": flags,
        "penalties": penalties,
        "meta": {"url": home.url, "title": home.title, "description": home.description, "h1": home.h1, "pages": [p.url for p in pages], "strictness": strictness},
    }

def analyze_url(url: str, industry: str, area: str, service: str, max_pages: int, strictness: str = "詳細診断") -> Tuple[Dict, List[PageData], Optional[str]]:
    try:
        target = normalize_url(url)
        home = fetch_page(target)
        links = select_important_links(home, max_pages)
        pages = [home]
        for link in links:
            try:
                time.sleep(0.25)
                pages.append(fetch_page(link))
            except Exception:
                pass
        return score_site(pages, industry, area, service, strictness), pages, None
    except Exception as e:
        return {}, [], str(e)


def make_comments(result: Dict) -> Dict[str, List[str]]:
    scores = result["scores"]
    flags = result["flags"]
    good = []
    issues = []
    actions = []

    for k, v in scores.items():
        if v >= 75:
            good.append(f"{k}は比較的整っています（{v}点）。")
        elif v < 55:
            issues.append(f"{k}が弱く、問い合わせ前の不安や離脱につながる可能性があります（{v}点）。")

    if not flags["has_price"]:
        actions.append("料金ページまたは費用目安を追加し、相談前の不安を下げる。")
    if not flags["has_case"]:
        actions.append("解決事例・お客様の声・対応実績を追加し、信頼材料を強化する。")
    if not flags["has_flow"]:
        actions.append("相談から完了までの流れを掲載し、依頼後のイメージを明確にする。")
    elif not flags.get("has_flow_detail", False):
        actions.append("相談の流れは掲載済みのため、各ステップに期間・必要書類・費用発生タイミングを補足して、依頼後のイメージをさらに明確にする。")
    if not flags["has_faq"]:
        actions.append("よくある質問を追加し、問い合わせ前の疑問を先回りして解消する。")
    elif not flags.get("has_faq_detail", False):
        actions.append("よくある質問は掲載済みのため、料金・期限・必要書類・依頼範囲など問い合わせ前に多い質問を増やす。")
    if not flags["has_phone"] or not flags["has_contact_link"] or not flags["has_form"]:
        actions.append("電話・フォーム・LINEなど複数の問い合わせ導線を目立つ位置に設置する。特に実フォームの存在を確認できる状態にする。")
    if flags.get("is_image_heavy"):
        actions.append("画像バナーに依存している訴求を、検索・AI・スマホで読み取りやすいHTMLテキストに置き換える。")
    if scores["ファーストビュー"] < 70:
        actions.append("トップの見出しを『地域名＋業務名＋相談メリット』の形に変更する。")

    if not good:
        good.append("現時点では大きく強い項目が少ないため、基本情報と導線の整備から始めるのがおすすめです。")
    if not issues:
        issues.append("大きな欠点は少ないですが、競合比較とCV計測でさらに改善余地を確認できます。")
    for reason, pts in result.get("penalties", []):
        issues.append(f"重要減点：{reason}（-{pts}点）。")

    if not actions:
        actions.append("競合上位サイトと比較し、料金・事例・CTAの見せ方をさらに強化する。")

    return {"good": good[:5], "issues": issues[:6], "actions": actions[:7]}


def sales_summary(result: Dict, comparison: Optional[Dict] = None) -> str:
    total = result["total"]
    rank = result["rank"]
    weak = sorted(result["scores"].items(), key=lambda x: x[1])[:3]
    weak_text = "、".join([f"{k}（{v}点）" for k, v in weak])
    penalty_text = ""
    if result.get("penalties"):
        penalty_text = " 重要減点項目は" + "、".join([f"{r}（-{p}点）" for r, p in result["penalties"][:4]]) + "です。"
    base = (
        f"総合評価は{rank}ランク / {total}点です。現在のHPは、問い合わせ獲得を目的とした実務基準で見ると、"
        f"特に{weak_text}に改善余地があります。"
        f"{penalty_text} 見込み客が相談前に感じる『費用・実績・流れ・相談しやすさ』の不安を解消することで、"
        "HPからの問い合わせ率改善が期待できます。まずはファーストビュー、料金情報、信頼材料、スマホ導線の順で改善することをおすすめします。"
    )
    if comparison and comparison.get("competitor_count", 0) > 0:
        diff = comparison["target_total"] - comparison["competitor_avg_total"]
        sign = "上回っています" if diff >= 0 else "下回っています"
        base += f" 競合平均との比較では、総合スコアが{abs(diff):.1f}点{sign}。差が大きい項目から改善すると、比較検討時に選ばれやすくなります。"
    return base


def build_comparison(target_result: Dict, competitor_results: List[Dict]) -> Dict:
    valid = [r for r in competitor_results if r]
    if not valid:
        return {"competitor_count": 0}
    categories = list(target_result["scores"].keys())
    avg_scores = {k: round(sum(r["scores"][k] for r in valid) / len(valid), 1) for k in categories}
    rows = []
    for k in categories:
        target = target_result["scores"][k]
        avg = avg_scores[k]
        diff = round(target - avg, 1)
        status = "優位" if diff >= 10 else "同等" if diff > -10 else "劣後"
        rows.append({"評価項目": k, "自社": target, "競合平均": avg, "差分": diff, "判定": status})
    total_avg = round(sum(r["total"] for r in valid) / len(valid), 1)
    return {
        "competitor_count": len(valid),
        "competitor_avg_total": total_avg,
        "target_total": target_result["total"],
        "rows": rows,
        "competitor_totals": [{"URL": r["meta"]["url"], "総合スコア": r["total"], "ランク": r["rank"]} for r in valid],
        "weak_vs_competitors": sorted(rows, key=lambda x: x["差分"])[:3],
    }


def pdf_p(text: str, style):
    safe = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe, style)


def generate_pdf_report(target_result: Dict, comments: Dict, comparison: Optional[Dict], industry: str, area: str, service: str) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlabがインストールされていません。requirements.txtを確認してください。")

    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="JPTitle", fontName="HeiseiKakuGo-W5", fontSize=20, leading=26, alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="JPSubTitle", fontName="HeiseiKakuGo-W5", fontSize=13, leading=18, spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="JPBody", fontName="HeiseiMin-W3", fontSize=9.5, leading=14))
    styles.add(ParagraphStyle(name="JPSmall", fontName="HeiseiMin-W3", fontSize=8, leading=11))

    story = []
    story.append(pdf_p("HP分析レポート", styles["JPTitle"]))
    story.append(pdf_p(f"作成日：{datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["JPSmall"]))
    story.append(Spacer(1, 6))

    summary_data = [
        ["分析URL", target_result["meta"]["url"]],
        ["業種", industry],
        ["対応エリア", area or "未指定"],
        ["狙いたい業務", service or "未指定"],
        ["総合評価", f"{target_result['rank']}ランク / {target_result['total']}点"],
    ]
    table = Table(summary_data, colWidths=[32 * mm, 138 * mm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "HeiseiMin-W3"),
        ("FONTNAME", (0, 0), (0, -1), "HeiseiKakuGo-W5"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF3F8")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    story.append(Spacer(1, 8))

    story.append(pdf_p("総評", styles["JPSubTitle"]))
    story.append(pdf_p(sales_summary(target_result, comparison), styles["JPBody"]))
    story.append(Spacer(1, 8))

    story.append(pdf_p("項目別スコア", styles["JPSubTitle"]))
    score_data = [["評価項目", "スコア"]] + [[k, f"{v}点"] for k, v in target_result["scores"].items()]
    score_table = Table(score_data, colWidths=[118 * mm, 28 * mm])
    score_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "HeiseiMin-W3"),
        ("FONTNAME", (0, 0), (-1, 0), "HeiseiKakuGo-W5"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F6F8B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(score_table)

    story.append(PageBreak())
    story.append(pdf_p("改善ポイント", styles["JPTitle"]))
    for title, items in [("良い点", comments["good"]), ("課題", comments["issues"]), ("優先改善案", comments["actions"])] :
        story.append(pdf_p(title, styles["JPSubTitle"]))
        for item in items:
            story.append(pdf_p("・" + item, styles["JPBody"]))
        story.append(Spacer(1, 4))

    if comparison and comparison.get("competitor_count", 0) > 0:
        story.append(PageBreak())
        story.append(pdf_p("競合比較", styles["JPTitle"]))
        story.append(pdf_p(f"競合{comparison['competitor_count']}サイトの平均スコア：{comparison['competitor_avg_total']}点", styles["JPBody"]))
        comp_data = [["評価項目", "自社", "競合平均", "差分", "判定"]] + [
            [r["評価項目"], r["自社"], r["競合平均"], r["差分"], r["判定"]] for r in comparison["rows"]
        ]
        comp_table = Table(comp_data, colWidths=[60 * mm, 23 * mm, 30 * mm, 25 * mm, 24 * mm])
        comp_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "HeiseiMin-W3"),
            ("FONTNAME", (0, 0), (-1, 0), "HeiseiKakuGo-W5"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F6F8B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(comp_table)
        story.append(Spacer(1, 8))
        story.append(pdf_p("競合平均との差が大きい改善候補", styles["JPSubTitle"]))
        for r in comparison["weak_vs_competitors"]:
            story.append(pdf_p(f"・{r['評価項目']}：競合平均より{abs(r['差分'])}点低い／判定：{r['判定']}", styles["JPBody"]))

    story.append(Spacer(1, 10))
    story.append(pdf_p("注意：このレポートはURL取得情報に基づく簡易診断です。実アクセス数・問い合わせ数・CVR・広告成果はGA4、Search Console、広告データ等との連携が必要です。", styles["JPSmall"]))
    doc.build(story)
    return buffer.getvalue()


st.set_page_config(page_title="HP分析ツール MVP", layout="wide")
st.title("HP分析ツール MVP")
st.caption("URLを入力すると、HPが問い合わせにつながる状態かを診断します。お客様に提示しやすいよう、初期設定は詳細診断です。競合URL比較とPDFレポート出力に対応。")

with st.sidebar:
    st.header("分析条件")
    url = st.text_input("HP URL", placeholder="https://example.com")
    industry = st.selectbox("業種", ["士業", "弁護士", "司法書士", "税理士", "行政書士", "その他"])
    area = st.text_input("対応エリア", placeholder="例：柏市")
    service = st.text_input("狙いたい業務", placeholder="例：相続登記")
    max_pages = st.slider("追加で取得する主要ページ数", 0, 8, 4)
    strictness = st.selectbox("診断レベル", ["詳細診断", "通常診断", "簡易診断"], index=0)
    st.markdown("---")
    competitor_urls_text = st.text_area("競合URL（任意・1行に1URL）", placeholder="https://competitor1.com\nhttps://competitor2.com", height=120)
    run = st.button("診断する", type="primary")

if run:
    if not url:
        st.error("URLを入力してください。")
        st.stop()

    with st.spinner("自社HPを取得・分析しています..."):
        result, pages, error = analyze_url(url, industry, area, service, max_pages, strictness)
    if error:
        st.error(f"分析に失敗しました: {error}")
        st.stop()

    competitor_urls = [x.strip() for x in competitor_urls_text.splitlines() if x.strip()]
    competitor_results = []
    competitor_errors = []
    if competitor_urls:
        with st.spinner("競合HPを取得・分析しています..."):
            for competitor_url in competitor_urls[:5]:
                comp_result, _, comp_error = analyze_url(competitor_url, industry, area, service, max_pages, strictness)
                if comp_error:
                    competitor_errors.append({"URL": competitor_url, "エラー": comp_error})
                else:
                    competitor_results.append(comp_result)
    comparison = build_comparison(result, competitor_results) if competitor_results else {"competitor_count": 0}
    comments = make_comments(result)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("総合スコア", f"{result['total']}点")
    c2.metric("判定ランク", result["rank"])
    c3.metric("分析ページ数", f"{len(pages)}ページ")
    if comparison.get("competitor_count", 0) > 0:
        c4.metric("競合平均との差", f"{result['total'] - comparison['competitor_avg_total']:.1f}点")
    else:
        c4.metric("診断レベル", strictness)

    st.subheader("項目別スコア")
    df = pd.DataFrame([{"評価項目": k, "スコア": v} for k, v in result["scores"].items()])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("評価項目"))

    st.subheader("総評")
    st.info(sales_summary(result, comparison))

    if result.get("penalties"):
        with st.expander("重要減点項目"):
            penalty_df = pd.DataFrame([{"減点理由": r, "減点": f"-{p}点"} for r, p in result["penalties"]])
            st.dataframe(penalty_df, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 良い点")
        for x in comments["good"]:
            st.write("- " + x)
    with col2:
        st.markdown("### 課題")
        for x in comments["issues"]:
            st.write("- " + x)
    with col3:
        st.markdown("### 優先改善案")
        for x in comments["actions"]:
            st.write("- " + x)

    if comparison.get("competitor_count", 0) > 0:
        st.subheader("競合URL比較")
        st.write(f"競合{comparison['competitor_count']}サイトの平均スコア：**{comparison['competitor_avg_total']}点**")
        comp_df = pd.DataFrame(comparison["rows"])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.markdown("### 競合と比べた優先改善候補")
        for r in comparison["weak_vs_competitors"]:
            st.write(f"- {r['評価項目']}：競合平均より{abs(r['差分'])}点低い（判定：{r['判定']}）")
        st.markdown("### 競合別スコア")
        st.dataframe(pd.DataFrame(comparison["competitor_totals"]), use_container_width=True, hide_index=True)

    if competitor_errors:
        with st.expander("取得できなかった競合URL"):
            st.dataframe(pd.DataFrame(competitor_errors), use_container_width=True, hide_index=True)

    st.subheader("PDFレポート出力")
    if REPORTLAB_AVAILABLE:
        try:
            pdf_bytes = generate_pdf_report(result, comments, comparison, industry, area, service)
            st.download_button(
                "PDFレポートをダウンロード",
                data=pdf_bytes,
                file_name="hp_analysis_report.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"PDF生成に失敗しました: {e}")
    else:
        st.warning("PDF出力にはreportlabが必要です。requirements.txtからインストールしてください。")

    st.subheader("取得できた基本情報")
    st.write("**title:**", result["meta"]["title"])
    st.write("**description:**", result["meta"]["description"])
    st.write("**h1:**", ", ".join(result["meta"]["h1"]) if result["meta"]["h1"] else "なし")
    st.write("**取得ページ:**")
    for p in result["meta"]["pages"]:
        st.write("- " + p)

    st.caption("注意：このMVPはルールベース診断です。実アクセス数・問い合わせ数・CVR・広告成果は外部連携が必要です。")

else:
    st.markdown("""
### 診断できる項目
- ファーストビュー
- 訴求力
- 信頼力
- 問い合わせ力
- 料金の分かりやすさ
- コンテンツの充実度
- SEO内部構造
- 技術・基本品質

### 追加済み機能
- 競合URLとの比較
- PDFレポート出力

### 今後追加しやすい機能
- 採点厳格化：料金・事例・プロフィール・問い合わせ導線・地域/業務訴求が弱い場合は重要減点
- PageSpeed Insights API連携
- Google Search Console / GA4連携
- OpenAI APIによる文章・訴求評価の高度化
- 士業別テンプレート診断
""")
