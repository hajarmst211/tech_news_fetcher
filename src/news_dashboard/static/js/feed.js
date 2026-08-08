const input = document.getElementById("search-input");
const hint = document.getElementById("search-hint");
const list = document.getElementById("feed-list");
const empty = document.getElementById("feed-empty");

let debounceTimer = null;

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str || "";
    return div.innerHTML;
}

function renderSentiment(sentiment, commentCount) {
    if (!commentCount) return "";
    const order = ["positive", "neutral", "negative"];
    const parts = order
        .filter((label) => sentiment[label])
        .map((label) => `<span class="sentiment-dot ${label}"></span>${sentiment[label]}`);
    if (!parts.length) return "";
    return `<div class="sentiment-bar">${parts.join(" ")}</div>`;
}

function renderCard(item) {
    const topics = (item.topics || [])
        .map((t) => `<span class="topic-chip">#${escapeHtml(t)}</span>`)
        .join("");
    return `
        <article class="item-card">
            <div class="item-card-top">
                <div class="item-title"><a href="${item.url}" target="_blank" rel="noopener">${escapeHtml(item.title)}</a></div>
                <div class="item-source">${escapeHtml(item.source_name)}</div>
            </div>
            <p class="item-summary">${escapeHtml(item.summary || "no summary available")}</p>
            <div class="item-meta">
                <div class="topic-chips">${topics}</div>
                ${renderSentiment(item.comment_sentiment || {}, item.comment_count)}
                <a class="item-link" href="${item.url}" target="_blank" rel="noopener">open link</a>
            </div>
        </article>
    `;
}

async function loadFeed(query) {
    empty.style.display = "block";
    empty.textContent = "searching...";
    const res = await fetch(`/api/feed?q=${encodeURIComponent(query)}`);
    const items = await res.json();

    if (!items.length) {
        list.innerHTML = "";
        empty.style.display = "block";
        empty.textContent = query
            ? "no items fetched today match that search"
            : "no items have been fetched today yet";
        hint.textContent = "";
        return;
    }

    empty.style.display = "none";
    list.innerHTML = items.map(renderCard).join("");
    hint.textContent = `${items.length} item${items.length === 1 ? "" : "s"} today`;
}

input.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => loadFeed(input.value.trim()), 250);
});

loadFeed("");
