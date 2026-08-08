const colors = {
    accent: "#e8a33d",
    text: "#e7e5e0",
    textDim: "#8b8d95",
    grid: "#2b2d33",
    positive: "#5fb87c",
    neutral: "#8b8d95",
    negative: "#d9695f",
};

Chart.defaults.color = colors.textDim;
Chart.defaults.font.family = "'JetBrains Mono', monospace";
Chart.defaults.font.size = 11;

let ngramChart, topicsChart, sentimentChart;

function baseBarOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { grid: { color: colors.grid }, ticks: { color: colors.textDim } },
            y: { grid: { color: colors.grid }, ticks: { color: colors.textDim }, beginAtZero: true },
        },
    };
}

async function loadNgrams(n) {
    const res = await fetch(`/api/ngrams?n=${n}`);
    const data = await res.json();
    const labels = data.map((d) => d.term);
    const weights = data.map((d) => d.weight);

    if (ngramChart) {
        ngramChart.data.labels = labels;
        ngramChart.data.datasets[0].data = weights;
        ngramChart.update();
        return;
    }

    ngramChart = new Chart(document.getElementById("ngram-chart"), {
        type: "bar",
        data: {
            labels,
            datasets: [{ data: weights, backgroundColor: colors.accent, borderRadius: 4 }],
        },
        options: baseBarOptions(),
    });
}

async function loadTopics() {
    const res = await fetch("/api/topics");
    const data = await res.json();
    topicsChart = new Chart(document.getElementById("topics-chart"), {
        type: "bar",
        data: {
            labels: data.map((d) => d.topic),
            datasets: [{ data: data.map((d) => d.count), backgroundColor: colors.accent, borderRadius: 4 }],
        },
        options: baseBarOptions(),
    });
}

async function loadTopicOptions() {
    const select = document.getElementById("topic-select");
    const res = await fetch("/api/topics/list");
    const topics = await res.json();
    topics.forEach((t) => {
        const opt = document.createElement("option");
        opt.value = t;
        opt.textContent = t;
        select.appendChild(opt);
    });
}

async function loadSentimentTrend(topic) {
    const res = await fetch(`/api/sentiment-trend?topic=${encodeURIComponent(topic)}`);
    const data = await res.json();

    const datasets = [
        { label: "positive", data: data.positive, borderColor: colors.positive, backgroundColor: colors.positive, tension: 0.3 },
        { label: "neutral", data: data.neutral, borderColor: colors.neutral, backgroundColor: colors.neutral, tension: 0.3 },
        { label: "negative", data: data.negative, borderColor: colors.negative, backgroundColor: colors.negative, tension: 0.3 },
    ];

    if (sentimentChart) {
        sentimentChart.data.labels = data.days;
        sentimentChart.data.datasets = datasets;
        sentimentChart.update();
        return;
    }

    sentimentChart = new Chart(document.getElementById("sentiment-chart"), {
        type: "line",
        data: { labels: data.days, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: colors.textDim } } },
            scales: {
                x: { grid: { color: colors.grid }, ticks: { color: colors.textDim } },
                y: { grid: { color: colors.grid }, ticks: { color: colors.textDim }, beginAtZero: true },
            },
        },
    });
}

document.querySelectorAll(".ngram-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".ngram-tab").forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        loadNgrams(tab.dataset.n);
    });
});

document.getElementById("topic-select").addEventListener("change", (e) => {
    loadSentimentTrend(e.target.value);
});

loadNgrams(1);
loadTopics();
loadTopicOptions();
loadSentimentTrend("");
