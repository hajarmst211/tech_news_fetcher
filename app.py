import json
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras

pd.set_option('display.max_colwidth', None)

load_dotenv()

st.set_page_config(page_title="Pipeline Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    .stApp > header { background-color: #0f172a !important; }
    .stat-card { background: #0f172a; border: 1px solid #334155; border-radius: 12px; padding: 1.25rem; }
    .stat-label { color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
    .stat-value { color: #f1f5f9; font-size: 1.875rem; font-weight: 700; font-family: monospace; margin-top: 0.25rem; }
    .stat-sub { color: #64748b; font-size: 0.75rem; margin-top: 0.25rem; }
    .card { background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 1.5rem; }
    .card-title { color: #f1f5f9; font-size: 1.125rem; font-weight: 700; margin-bottom: 1rem; }
    .tag { background: #1e293b; border: 1px solid #334155; color: #cbd5e1; padding: 0.125rem 0.5rem; border-radius: 4px; font-size: 0.625rem; font-family: monospace; display: inline-block; margin: 0.125rem; }
    .tag-lg { background: #1e293b; border: 1px solid #334155; border-radius: 9999px; padding: 0.25rem 0.625rem; font-size: 0.75rem; display: inline-block; margin: 0.125rem; }
    .metric-box { border-radius: 8px; padding: 1rem; text-align: center; }
    .stTextInput>div>div>input { background-color: #1e293b; border-color: #334155; color: #e2e8f0; }
    div[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
    div[data-testid="stSidebar"] .st-emotion-cache-1v7f65g { color: #e2e8f0; }
    .stDataFrame { background: #0f172a; }
</style>
""", unsafe_allow_html=True)


def get_db():
    if "db" not in st.session_state:
        try:
            conn = psycopg2.connect(os.environ["DATABASE_URL"])
            st.session_state.db = conn
        except Exception:
            st.session_state.db = None
    return st.session_state.db


def load_stats():
    try:
        with open("stats.json") as f:
            return json.load(f)
    except Exception:
        return None


def stat_card(label, value, sub=None):
    sub_html = f'<p class="stat-sub">{sub}</p>' if sub else ""
    return f'<div class="stat-card"><p class="stat-label">{label}</p><p class="stat-value">{value}</p>{sub_html}</div>'


def bar_segment(val, total, color):
    if val == 0 or total == 0:
        return ""
    pct = val / total * 100
    return f'<div style="background:{color};height:0.75rem;width:{pct:.1f}%"></div>'


def render_source_tags(sources, fallback="None"):
    if not sources or len(sources) == 0:
        return f'<span style="color:#475569;font-style:italic;font-size:0.75rem">{fallback}</span>'
    tags = ""
    for s in sources:
        name = s if isinstance(s, str) else (s.get("name") or s.get("source") or "")
        count = f" ({s['count']})" if isinstance(s, dict) and "count" in s else ""
        tags += f'<span class="tag">{name}{count}</span>'
    return f'<div style="display:flex;flex-wrap:wrap;gap:0.25rem;margin-top:0.25rem">{tags}</div>'


# ─── Page: Dashboard ───────────────────────────────────────────────
def dashboard_page():
    st.header("Pipeline Dashboard", divider=False)
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    data = load_stats()
    if data is None:
        st.warning("Could not load stats.json. Run `python3 generate_stats.py` first.")
        st.stop()

    d = data
    sc = d["summary_content_breakdown"]
    total_items = d["db_counts"]["items"]
    imp = d["summarizer_impact"]

    neither_sources = d.get("neither_sources", []) or []
    content_no_summary_sources = (
        d.get("content_no_summary_sources") or
        d.get("no_summary_sources") or
        d.get("content_only_sources") or
        []
    )

    # Row 1: Top-level pipeline counts
    cols = st.columns(4)
    with cols[0]:
        st.markdown(stat_card("Sources (total)", d["sources"]["total"]), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(stat_card("Items in DB", d["db_counts"]["items"], f"{d.get('items_with_summary',0)} with summaries"), unsafe_allow_html=True)
    with cols[2]:
        cs = f"{d.get('items_with_comments',0)} items have comments" if d.get("items_with_comments", 0) > 0 else ""
        st.markdown(stat_card("Comments in DB", d["db_counts"]["comments"], cs), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(stat_card("Vulnerabilities", d["db_counts"]["vulnerabilities"]), unsafe_allow_html=True)

    # Row 2: Sources detail
    with st.container():
        st.markdown(f'<div class="card"><h2 class="card-title">Sources</h2>', unsafe_allow_html=True)
        type_colors = {
            "devto": "bg-indigo-500", "arxiv": "bg-emerald-500", "rss": "bg-amber-500",
            "hn": "bg-rose-500", "github_release": "bg-cyan-500", "nvd": "bg-red-500"
        }
        src_cols = st.columns(6)
        for i, (t, cnt) in enumerate(sorted(d["sources"]["type_counts"].items(), key=lambda x: x[1], reverse=True)):
            with src_cols[i % 6]:
                st.markdown(f'<div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:0.75rem;text-align:center"><p style="font-size:1.125rem;font-weight:700;color:#f1f5f9;font-family:monospace">{cnt}</p><p style="font-size:0.75rem;color:#64748b">{t}</p></div>', unsafe_allow_html=True)

        st.markdown('<div style="max-height:16rem;overflow-y:auto">', unsafe_allow_html=True)
        src_data = [
            {
                "Source": s["name"],
                "Type": s["type"],
                "Category": s.get("category") or "—",
                "Items": s["items"],
                "Status": s.get("status") or "unknown",
                "Code": s.get("status_code") if s.get("status_code") is not None else "—"
            }
            for s in d["sources"]["list"] if s["items"] > 0
        ]
        if src_data:
            st.dataframe(src_data, use_container_width=True, height=256, hide_index=True,
                         column_config={"Status": st.column_config.TextColumn(width="small")})
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Row 3: Summary / Content Breakdown
    with st.container():
        st.markdown(f'<div class="card"><h2 class="card-title">Summary / Content Breakdown</h2>', unsafe_allow_html=True)
        cols = st.columns(4)
        labels = [
            ("summary AND content", "#10b981", "#065f46", sc.get("summary AND content", 0)),
            ("summary, no content", "#f59e0b", "#78350f", sc.get("summary, no content", 0)),
            ("no summary, content", "#3b82f6", "#1e3a5f", sc.get("no summary, content", 0)),
            ("no summary, no content", "#64748b", "#1e293b", sc.get("no summary, no content", 0)),
        ]
        for idx, (label, border, bg, val) in enumerate(labels):
            with cols[idx]:
                st.markdown(f'<div class="metric-box" style="background:{bg}20;border:1px solid {border}40"><p style="font-size:1.5rem;font-weight:700;font-family:monospace;color:{border}">{val}</p><p style="font-size:0.75rem;margin-top:0.25rem;opacity:0.7;color:{border}">{label}</p></div>', unsafe_allow_html=True)

        bar_html = '<div style="width:100%;background:#1e293b;border-radius:9999px;height:0.75rem;overflow:hidden;display:flex">'
        bar_html += bar_segment(sc.get("summary AND content", 0), total_items, "#10b981")
        bar_html += bar_segment(sc.get("summary, no content", 0), total_items, "#f59e0b")
        bar_html += bar_segment(sc.get("no summary, content", 0), total_items, "#3b82f6")
        bar_html += bar_segment(sc.get("no summary, no content", 0), total_items, "#475569")
        bar_html += '</div><div style="display:flex;gap:1rem;margin-top:0.5rem"><span style="font-size:0.625rem;color:#64748b">● summary+content</span><span style="font-size:0.625rem;color:#64748b">● summary only</span><span style="font-size:0.625rem;color:#64748b">● content only</span><span style="font-size:0.625rem;color:#64748b">● neither</span></div>'
        st.markdown(bar_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 4: Summarizer Impact
    with st.container():
        st.markdown(f'<div class="card"><h2 class="card-title">Summarizer Impact</h2>', unsafe_allow_html=True)
        before, after = st.columns(2)

        with before:
            st.markdown('<div style="background:rgba(30,41,59,0.5);border:1px solid #334155;border-radius:12px;padding:1.25rem">', unsafe_allow_html=True)
            st.markdown('<p style="font-size:0.875rem;font-weight:600;color:#94a3b8;text-transform:uppercase;margin-bottom:1rem"><span style="display:inline-block;width:0.5rem;height:0.5rem;border-radius:50%;background:#64748b;margin-right:0.5rem"></span>Before Summarizer</p>', unsafe_allow_html=True)
            b = imp["before"]
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(51,65,85,0.5)"><span style="color:#94a3b8;font-size:0.875rem">Content, no summary</span><span style="font-size:1.125rem;font-weight:700;color:#93c5fd;font-family:monospace">{b["content_no_summary"]}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(51,65,85,0.5)"><span style="color:#94a3b8;font-size:0.875rem">Summary, no content</span><span style="font-size:1.125rem;font-weight:700;color:#fcd34d;font-family:monospace">{b["summary_no_content"]}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:0.5rem 0"><span style="color:#94a3b8;font-size:0.875rem">No summary, no content</span><span style="font-size:1.125rem;font-weight:700;color:#94a3b8;font-family:monospace">{b["neither"]}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="margin-top:0.5rem;padding-top:0.25rem;border-top:1px solid #0f172a"><p style="font-size:0.625rem;color:#64748b;text-transform:uppercase;font-weight:600">Sources with no content/summary:</p>{render_source_tags(neither_sources, "No empty sources")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="margin-top:0.75rem;padding-top:0.75rem;border-top:1px solid #334155"><p style="font-size:0.75rem;color:#64748b">{b["content_no_summary"]} items were candidates for summarization (had full article content but no summary).</p></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with after:
            st.markdown('<div style="background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.2);border-radius:12px;padding:1.25rem">', unsafe_allow_html=True)
            st.markdown('<p style="font-size:0.875rem;font-weight:600;color:#6ee7b7;text-transform:uppercase;margin-bottom:1rem"><span style="display:inline-block;width:0.5rem;height:0.5rem;border-radius:50%;background:#10b981;margin-right:0.5rem"></span>After Summarizer</p>', unsafe_allow_html=True)
            a = imp["after"]
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(51,65,85,0.5)"><span style="color:#94a3b8;font-size:0.875rem">Summary AND content</span><span style="font-size:1.125rem;font-weight:700;color:#6ee7b7;font-family:monospace">{a["summary_and_content"]}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="padding:0.5rem 0;border-bottom:1px solid rgba(51,65,85,0.5)"><div style="display:flex;justify-content:space-between"><span style="color:#94a3b8;font-size:0.875rem">Content, no summary</span><span style="font-size:1.125rem;font-weight:700;color:#93c5fd;font-family:monospace">{a["content_no_summary"]}</span></div><div style="margin-top:0.5rem;padding-top:0.25rem;border-top:1px solid #334155"><p style="font-size:0.625rem;color:#64748b;text-transform:uppercase;font-weight:600">Remaining sources with content but no summary:</p>{render_source_tags(content_no_summary_sources, "No remaining sources")}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(51,65,85,0.5)"><span style="color:#94a3b8;font-size:0.875rem">Summary, no content</span><span style="font-size:1.125rem;font-weight:700;color:#fcd34d;font-family:monospace">{a["summary_no_content"]}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:0.5rem 0"><span style="color:#94a3b8;font-size:0.875rem">No summary, no content</span><span style="font-size:1.125rem;font-weight:700;color:#94a3b8;font-family:monospace">{a["neither"]}</span></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Row 5: Items by source type & top sources
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f'<div class="card"><h2 class="card-title">Items by Source Type</h2>', unsafe_allow_html=True)
        items_by_type = d.get("items_by_source_type", {})
        sorted_types = sorted(items_by_type.items(), key=lambda x: x[1], reverse=True)
        max_val = max((v for _, v in sorted_types), default=1)
        type_bar_colors = {
            "devto": "#6366f1", "arxiv": "#10b981", "rss": "#f59e0b",
            "hn": "#f43f5e", "github_release": "#06b6d4", "nvd": "#ef4444"
        }
        for t, cnt in sorted_types:
            bar_color = type_bar_colors.get(t, "#64748b")
            bar_w = (cnt / max_val * 100) if max_val > 0 else 0
            st.markdown(f'<div style="margin-bottom:0.75rem"><div style="display:flex;justify-content:space-between;font-size:0.875rem;margin-bottom:0.25rem"><span style="color:#94a3b8">{t}</span><span style="color:#f1f5f9;font-family:monospace;font-weight:700">{cnt}</span></div><div style="width:100%;background:#1e293b;border-radius:9999px;height:0.75rem;overflow:hidden"><div style="background:{bar_color};height:0.75rem;border-radius:9999px;width:{bar_w:.0f}%"></div></div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown(f'<div class="card"><h2 class="card-title">Top Sources by Items</h2>', unsafe_allow_html=True)
        for s in d.get("top_sources_by_items", []):
            name = s["name"][:35] + "..." if len(s["name"]) > 35 else s["name"]
            st.markdown(f'<div style="display:flex;align-items:center;justify-content:space-between;background:rgba(30,41,59,0.5);border:1px solid #334155;border-radius:8px;padding:0.5rem 1rem;margin-bottom:0.5rem"><span style="font-size:0.875rem;color:#94a3b8;font-family:monospace;overflow:hidden;text-overflow:ellipsis">{name}</span><span style="font-size:0.875rem;font-weight:700;color:#f1f5f9;font-family:monospace;margin-left:0.5rem">{s["count"]}</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 6: Comments
    with st.container():
        st.markdown(f'<div class="card"><h2 class="card-title">Comments</h2>', unsafe_allow_html=True)
        if d["db_counts"]["comments"] > 0 and d.get("comments_by_source"):
            comment_data = [{"Source": c["source"], "Comment Count": c["count"]} for c in d["comments_by_source"]]
            st.dataframe(comment_data, use_container_width=True, height=256, hide_index=True)
        else:
            st.markdown('<p style="text-align:center;padding:2rem 0;color:#64748b;font-size:0.875rem">No comments in database yet.</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 7: Themes
    with st.container():
        themes = d.get("themes", {})
        st.markdown(f'<div class="card"><h2 class="card-title">Extracted Themes</h2>', unsafe_allow_html=True)

        tag_html = '<div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-bottom:1.5rem">'
        for t in themes.get("list", [])[:80]:
            size = "font-size:1rem" if t["count"] >= 5 else ("font-size:0.875rem" if t["count"] >= 3 else "font-size:0.75rem")
            if t["count"] >= 5:
                style = "background:rgba(99,102,241,0.15);color:#e0e7ff;border:1px solid rgba(99,102,241,0.4)"
            elif t["count"] >= 3:
                style = "background:rgba(16,185,129,0.1);color:#a7f3d0;border:1px solid rgba(16,185,129,0.3)"
            else:
                style = "background:#1e293b;color:#cbd5e1;border:1px solid #334155"
            tag_html += f'<span class="tag-lg" style="{style};{size}">{t["theme"]} <span style="opacity:0.6">({t["count"]})</span></span>'
        total_distinct = themes.get("total_distinct", 0)
        if total_distinct > 80:
            tag_html += f'<span style="font-size:0.75rem;color:#64748b;padding:0.25rem 0.5rem">+{total_distinct - 80} more...</span>'
        tag_html += "</div>"
        st.markdown(tag_html, unsafe_allow_html=True)

        search = st.text_input("Search themes...", placeholder="Filter themes...", label_visibility="collapsed", key="theme_search")
        theme_list = themes.get("list", [])
        if search:
            theme_list = [t for t in theme_list if search.lower() in t["theme"].lower()]

        if theme_list:
            theme_data = [{"Theme": t["theme"], "Count": t["count"]} for t in theme_list]
            st.dataframe(theme_data, use_container_width=True, height=320, hide_index=True)
        else:
            st.markdown('<p style="color:#64748b;font-size:0.875rem;text-align:center;padding:2rem">No themes found.</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ─── Page: Comments ────────────────────────────────────────────────
def comments_page():
    st.header("Comment Sentiment Explorer", divider=False)

    db = get_db()
    if db is None:
        st.warning("Database not available. Check your DATABASE_URL in .env")
        st.stop()

    cur = db.cursor()

    cur.execute("""
        SELECT DISTINCT s.name
        FROM comments c
        JOIN items i ON c.item_id = i.id
        JOIN sources s ON i.source_id = s.id
        WHERE c.sentiment_label IS NOT NULL
        ORDER BY s.name
    """)
    sources = [row[0] for row in cur.fetchall()]

    if not sources:
        st.info("No comments with sentiment data found.")
        cur.close()
        return

    selected_source = st.selectbox("Select comment source", sources)

    cur.execute("""
        SELECT c.body_text, c.sentiment_label, c.sentiment_score
        FROM comments c
        JOIN items i ON c.item_id = i.id
        JOIN sources s ON i.source_id = s.id
        WHERE s.name = %s AND c.sentiment_label IS NOT NULL
        ORDER BY c.published_at
    """, (selected_source,))
    rows = cur.fetchall()
    cur.close()

    if not rows:
        st.info("No comments found for this source.")
        return

    st.markdown(f"**{len(rows)}** comments")

    color_map = {"positive": "#10b981", "neutral": "#f59e0b", "negative": "#ef4444"}

    for body, label, score in rows:
        c = color_map.get(label, "#64748b")
        display = body if body else "(empty)"
        st.markdown(
            f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:0.75rem;margin-bottom:0.5rem">'
            f'<p style="font-size:1.0rem;color:#cbd5e1;margin-bottom:0.5rem;line-height:1.5">{display}</p>'
            f'<div style="display:flex;gap:1rem;font-size:0.75rem">'
            f'<span style="color:{c};font-weight:600">{label.upper()}</span>'
            f'<span style="color:#64748b">{(score or 0):.3f}</span>'
            f'</div></div>',
            unsafe_allow_html=True
        )


# ─── Page: Theme Summaries ─────────────────────────────────────────
def theme_summaries_page():
    st.header("Theme Summary Explorer", divider=False)

    db = get_db()
    if db is None:
        st.warning("Database not available. Check your DATABASE_URL in .env")
        st.stop()

    cur = db.cursor()

    cur.execute("""
        SELECT DISTINCT i.theme
        FROM items i
        JOIN sources s ON i.source_id = s.id
        WHERE i.theme IS NOT NULL AND TRIM(i.theme) != ''
          AND s.source_type NOT IN ('nvd', 'github_release')
        ORDER BY i.theme
    """)
    themes = [row[0] for row in cur.fetchall()]

    if not themes:
        st.info("No themes found.")
        cur.close()
        return

    selected_theme = st.selectbox("Select a theme", themes)

    cur.execute("""
        SELECT i.id, i.title, i.summary, s.name
        FROM items i
        JOIN sources s ON i.source_id = s.id
        WHERE i.theme = %s
          AND s.source_type NOT IN ('nvd', 'github_release')
        ORDER BY i.published_at DESC
    """, (selected_theme,))
    rows = cur.fetchall()

    if not rows:
        st.info("No items found for this theme.")
        cur.close()
        return

    item_ids = [r[0] for r in rows]

    cur.execute("""
        SELECT c.item_id, c.body_text, c.sentiment_label, c.sentiment_score
        FROM comments c
        WHERE c.item_id = ANY(%s)
        ORDER BY c.published_at
    """, (item_ids,))
    comment_rows = cur.fetchall()
    cur.close()

    comments_by_item: dict[int, list[tuple[str, str | None, float | None]]] = {}
    for item_id, body, label, score in comment_rows:
        comments_by_item.setdefault(item_id, []).append((body, label, score))

    st.markdown(f"**{len(rows)}** items")

    sentiment_color = {"positive": "#10b981", "neutral": "#f59e0b", "negative": "#ef4444"}

    for item_id, title, summary, source in rows:
        card = (
            f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:1rem;margin-bottom:0.75rem">'
            f'<div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:0.5rem">'
            f'<p style="font-size:1rem;font-weight:600;color:#e0e7ff">{title}</p>'
            f'<span style="font-size:0.625rem;color:#64748b;font-family:monospace;background:#1e293b;padding:0.125rem 0.5rem;border-radius:4px;white-space:nowrap;margin-left:0.5rem">{source}</span>'
            f'</div>'
            f'<p style="font-size:1.0rem;color:#94a3b8;line-height:1.5">{(summary or "*No summary available*")}</p>'
        )

        item_comments = comments_by_item.get(item_id)
        if item_comments:
            comments_html = ""
            for body, label, score in item_comments:
                c = sentiment_color.get(label, "#64748b")
                display = body if body else "(empty)"
                comments_html += (
                    f'<div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:0.5rem;margin-top:0.5rem">'
                    f'<p style="font-size:0.8125rem;color:#cbd5e1;margin-bottom:0.25rem">{display}</p>'
                    f'<div style="display:flex;gap:1rem;font-size:0.7rem">'
                    f'<span style="color:{c};font-weight:600">{label.upper() if label else "N/A"}</span>'
                    f'<span style="color:#64748b">{(score or 0):.3f}</span>'
                    f'</div></div>'
                )
            card += f'<div style="margin-top:0.75rem;padding-top:0.75rem;border-top:1px solid #334155"><p style="font-size:0.75rem;color:#64748b;margin-bottom:0.5rem;font-weight:600">Comments ({len(item_comments)})</p>{comments_html}</div>'

        card += "</div>"
        st.markdown(card, unsafe_allow_html=True)


# ─── Navigation ────────────────────────────────────────────────────
page = st.sidebar.selectbox("Page", [
    "📊  Dashboard",
    "💬  Comments",
    "🏷️  Theme Summaries",
])

if "Dashboard" in page:
    dashboard_page()
elif "Comments" in page:
    comments_page()
elif "Theme Summaries" in page:
    theme_summaries_page()
