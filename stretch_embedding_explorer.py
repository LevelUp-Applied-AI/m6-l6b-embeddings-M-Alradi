"""
Module 6 Week B — Stretch: Embedding Space Explorer

Creates:
1. 2D visualization of 200 GloVe word embeddings (t-SNE)
2. 2D visualization of 20 BBC news DistilBERT embeddings (PCA)

Outputs:
- word_embedding_plot.png
- doc_embedding_plot.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from transformers import AutoTokenizer, AutoModel

from utils import load_glove, text_to_glove, extract_bert_embedding

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. WORD EMBEDDING SETUP (200 words, 5 categories)
word_categories = {
    "geopolitics_and_entities": [
        "sovereignty", "diplomacy", "parliament", "ambassador", "sanctions", "territory", "republic", "ministry", "election", "policy",
        "boundary", "alliance", "treaty", "governance", "citizenship", "metropolis", "province", "federation", "colony", "autonomy",
        "washington", "beijing", "london", "moscow", "berlin", "paris", "tokyo", "brussels", "cairo", "sydney",
        "voter", "senate", "congress", "capitalism", "communism", "democrat", "monarchy", "embassy", "border", "summit"
    ],
    "economics_and_finance": [
        "portfolio", "dividend", "liquidity", "inflation", "revenue", "deficit", "currency", "commodity", "transaction", "equity",
        "bankruptcy", "investment", "mortgage", "liability", "capital", "banking", "finance", "fiscal", "retail", "marketing",
        "consumer", "merchant", "inventory", "exchange", "startup", "monopoly", "subsidiary", "contract", "enterprise", "profit",
        "taxation", "interest", "budget", "pension", "insurance", "shareholder", "audit", "wealth", "asset", "salary"
    ],
    "science_and_nature": [
        "organism", "molecule", "evolution", "photosynthesis", "gravity", "galaxy", "ecosystem", "bacteria", "geology", "physics",
        "atmosphere", "nitrogen", "velocity", "membrane", "species", "habitat", "glacier", "volcano", "asteroid", "mutation",
        "laboratory", "element", "nebula", "quantum", "electron", "nucleus", "genetics", "fossil", "spectrum", "enzyme",
        "climate", "protein", "energy", "tide", "forest", "desert", "wildlife", "ocean", "biomass", "horizon"
    ],
    "technology_and_digital": [
        "algorithm", "encryption", "database", "processor", "bandwidth", "interface", "protocol", "software", "hardware", "server",
        "network", "wireless", "broadband", "automation", "robotics", "artificial", "intelligence", "computing", "silicon", "browser",
        "platform", "application", "firewall", "cyber", "storage", "cloud", "digital", "pixel", "circuit", "sensor",
        "programming", "mobile", "virtual", "online", "ethernet", "malware", "update", "memory", "binary", "hacker"
    ],
    "emotions_and_cognition": [
        "euphoria", "melancholy", "nostalgia", "resentment", "empathy", "anxiety", "courage", "distrust", "affection", "hostility",
        "serenity", "despair", "passion", "jealousy", "confusion", "curiosity", "gratitude", "contempt", "remorse", "ambition",
        "perception", "intellect", "intuition", "consciousness", "memory", "wisdom", "instinct", "belief", "thought", "emotion",
        "happiness", "sadness", "terror", "surprise", "disgust", "shame", "kindness", "bravery", "boredom", "wonder"
    ]
}

words = []
labels = []

for cat, ws in word_categories.items():
    words.extend(ws)
    labels.extend([cat] * len(ws))


# 2. LOAD GLOVE & BUILD WORD VECTORS
print("Loading GloVe embeddings...")
glove = load_glove("data/glove_50k_50d.txt")

print("Building word embeddings...")
X_words = np.array([
    text_to_glove(w, glove) for w in words
])


# 3. DIMENSIONALITY REDUCTION (WORDS → t-SNE)
print("Running t-SNE on word embeddings...")
tsne = TSNE(
    n_components=2,
    perplexity=20,
    random_state=42,
    init="pca"
)

X_words_2d = tsne.fit_transform(X_words)


# ── 4. WORD VISUALIZATION ────────────────────────────────────────────────────
print("Plotting word embeddings...")

# Palette — one distinct color per category
WORD_COLORS = {
    "science_and_nature":       "#7F77DD",   # purple
    "emotions_and_cognition":   "#EF9F27",   # amber
    "economics_and_finance":    "#378ADD",   # blue
    "geopolitics_and_entities": "#E24B4A",   # red
    "technology_and_digital":   "#3aab6d",   # green
}

CATEGORY_LABELS = {
    "science_and_nature":       "Science & nature",
    "emotions_and_cognition":   "Emotions & cognition",
    "economics_and_finance":    "Economics & finance",
    "geopolitics_and_entities": "Geopolitics & entities",
    "technology_and_digital":   "Technology & digital",
}

# Words to annotate — one or two strong representatives per cluster
# plus any notable outliers visible in the plot
ANNOTATE_WORDS = {
    "sovereignty", "diplomacy", "beijing", "brussels", "monarchy",
    "capitalism", "summit", "senate",
    "portfolio", "liquidity", "deficit", "fiscal", "shareholder", "asset",
    "bacteria", "mutation", "nebula", "forest", "membrane", "quantum",
    "broadband", "programming", "algorithm", "silicon", "hacker", "storage",
    "contempt", "despair", "consciousness", "nostalgia", "anxiety", "boredom",
}

fig, ax = plt.subplots(figsize=(14, 9))
fig.patch.set_facecolor("#f9f9f9")
ax.set_facecolor("#f9f9f9")

for i, word in enumerate(words):
    x, y   = X_words_2d[i]
    cat    = labels[i]
    color  = WORD_COLORS[cat]

    ax.scatter(x, y, color=color, alpha=0.75, s=38,
               linewidths=0.4, edgecolors="white", zorder=2)

    if word in ANNOTATE_WORDS:
        ax.annotate(
            word,
            xy=(x, y),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
            fontweight="semibold",
            zorder=3,
        )

# Legend
legend_handles = [
    mpatches.Patch(facecolor=WORD_COLORS[cat], label=CATEGORY_LABELS[cat])
    for cat in WORD_COLORS
]
ax.legend(
    handles=legend_handles,
    title="Semantic category",
    title_fontsize=9,
    fontsize=8.5,
    loc="upper left",
    framealpha=0.85,
    edgecolor="#cccccc",
)

ax.set_title("GloVe Word Embedding Space (t-SNE, perplexity=20)",
             fontsize=13, fontweight="semibold", pad=12)
ax.set_xlabel("Dimension 1", fontsize=10, color="#555")
ax.set_ylabel("Dimension 2", fontsize=10, color="#555")
ax.tick_params(colors="#888", labelsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#cccccc")
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#cccccc")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "word_embedding_plot.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 5. BBC NEWS DOCUMENT SELECTION (20 articles)
print("Loading BBC dataset...")
df = pd.read_csv("data/bbc_news.csv")

docs = []
doc_labels = []

for cat in df["category"].unique():
    subset = df[df["category"] == cat].head(4)  # 4 per category
    docs.extend(subset["text"].tolist())
    doc_labels.extend([cat] * len(subset))


# 6. DISTILBERT EMBEDDINGS
print("Loading DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()

print("Extracting document embeddings...")

X_docs = np.array([
    extract_bert_embedding(text, tokenizer, model)
    for text in docs
])


# 7. DIMENSIONALITY REDUCTION (DOCS → PCA)
print("Running PCA on document embeddings...")

pca = PCA(n_components=2)
X_docs_2d = pca.fit_transform(X_docs)


# ── 8. DOCUMENT VISUALIZATION ────────────────────────────────────────────────
print("Plotting document embeddings...")

DOC_COLORS = {
    "business":      "#E24B4A",   # red
    "entertainment": "#378ADD",   # blue
    "politics":      "#3aab6d",   # green
    "sport":         "#7F77DD",   # purple
    "tech":          "#EF9F27",   # amber
}

DOC_MARKERS = {
    "business":      "o",
    "entertainment": "s",
    "politics":      "^",
    "sport":         "D",
    "tech":          "P",
}

fig, ax = plt.subplots(figsize=(13, 8))
fig.patch.set_facecolor("#f9f9f9")
ax.set_facecolor("#f9f9f9")

# Plot each point with category color + unique marker
for i, text in enumerate(docs):
    x, y  = X_docs_2d[i]
    cat   = doc_labels[i]
    color  = DOC_COLORS.get(cat, "#888888")
    marker = DOC_MARKERS.get(cat, "o")

    ax.scatter(x, y,
               color=color, marker=marker,
               s=120, alpha=0.88,
               linewidths=0.6, edgecolors="white",
               zorder=2)

    # Short label: "category: first 22 chars"
    short = f"{text[:22].strip()}…"
    ax.annotate(
        short,
        xy=(x, y),
        xytext=(6, 4),
        textcoords="offset points",
        fontsize=7,
        color=color,
        zorder=3,
    )

# Variance explained
var = pca.explained_variance_ratio_
ax.set_xlabel(f"PC1  ({var[0]*100:.1f}% variance)", fontsize=10, color="#555")
ax.set_ylabel(f"PC2  ({var[1]*100:.1f}% variance)", fontsize=10, color="#555")

# Legend — color + marker shape
legend_handles = [
    Line2D([0], [0],
           marker=DOC_MARKERS[cat],
           color="w",
           markerfacecolor=DOC_COLORS[cat],
           markeredgecolor="white",
           markersize=9,
           label=cat.capitalize())
    for cat in DOC_COLORS
]
ax.legend(
    handles=legend_handles,
    title="BBC category",
    title_fontsize=9,
    fontsize=8.5,
    loc="upper left",
    framealpha=0.85,
    edgecolor="#cccccc",
)

ax.set_title("BBC News Embedding Space (DistilBERT + PCA)",
             fontsize=13, fontweight="semibold", pad=12)
ax.tick_params(colors="#888", labelsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#cccccc")
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#cccccc")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "doc_embedding_plot.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()


print("Done.")
print("Saved:")
print("  - output/word_embedding_plot.png")
print("  - output/doc_embedding_plot.png")