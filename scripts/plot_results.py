"""Figures for the results chapter.

    python scripts/plot_results.py                  # test split -> results/figures/
    python scripts/plot_results.py --split dev --out my_figures

Reads the tables in results/summary/ and the predictions in results/predictions/
(through score.py). Loads no model and makes no network call. Needs matplotlib,
which `python install.py --run` installs.

Writes each figure as PNG and PDF, and the numbers behind the three figures
that are not already a summary table as CSV next to them:

  fig1_overall              exact match per configuration, place-bootstrap interval
  fig2_condition_heatmap    exact match, configuration x condition
  fig3_noise                clean post vs demonym distractor vs entity distractor
  fig4_error_layers         stage at which a post was lost
  fig5_wrong_answers        what the wrong answer was, per condition
  fig6_prominence           exact match by place and rival prominence
  fig7_answer_precision     share of posts answered vs precision of the answers
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from score import (attach_split, config_order, containment, find_prediction_files,  # noqa: E402
                   load_predictions, row_order, score_record)

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_DIR = REPO_ROOT / "results" / "summary"
DEFAULT_OUT = REPO_ROOT / "results" / "figures"
DATASET_PATH = REPO_ROOT / "generation" / "synthetic" / "output" / "synthetic_dataset.csv"

LABEL = {
    "majority_class": "Majority class",
    "population_wikidata": "Population baseline",
    "geoparser_irchel": "Irchel Geoparser",
    "retrieve_rerank": "Search + rerank",
    "refined": "ReFinED",
    "blink": "BLINK",
    "dense_retrieval": "Dense retrieval",
    "gemma4": "Gemma 4 (LLM only)",
    "qwen3_vl": "Qwen3-VL (LLM only)",
    "grounded_gemma": "Gemma over dense candidates",
    "grounded_gemma_linking": "Gemma over ReFinED candidates",
}
# The palette every figure draws from.
TERRA_COTTA, CORAL, INDIAN_YELLOW, GRANDIS = "#C44B56", "#F2897B", "#F3A155", "#F5D687"
OLIVE, HAZEL, GUM_LEAF, AMETHYST = "#B9BA75", "#6C7F54", "#B3CFC3", "#A392AF"
DEEP_GUM_LEAF = "#6F9C8A"           # a darker Gum Leaf, the one colour added to the eight

# One colour per configuration, the same in every figure. Neighbouring colours
# are one kind of method: grey baselines, amethyst geoparser, gum leaf
# retrieval, coral and terra cotta entity linkers, yellows LLM alone, greens
# LLM over candidates. Line charts also vary the marker, for colour-blind
# readers and for print.
COLOUR = {
    "majority_class": "#c9c9c9",
    "population_wikidata": "#7f7f7f",
    "geoparser_irchel": AMETHYST,
    "retrieve_rerank": GUM_LEAF,
    "dense_retrieval": DEEP_GUM_LEAF,
    "refined": CORAL,
    "blink": TERRA_COTTA,
    "gemma4": GRANDIS,
    "qwen3_vl": INDIAN_YELLOW,
    "grounded_gemma": OLIVE,
    "grounded_gemma_linking": HAZEL,
}
# One outcome palette for the stacked figures: green is correct (the dark end
# of the heatmap's scale), amethyst is no answer, and the wrong answers run
# from gum leaf through the yellows to terra cotta.
CORRECT, NO_ANSWER = HAZEL, AMETHYST
DARK_FILLS = {HAZEL, TERRA_COTTA}
HEATMAP = LinearSegmentedColormap.from_list("accuracy", ["#fbfbf7", GUM_LEAF, HAZEL])
BAR_HEIGHT = 0.62
MARKER = {"population_wikidata": "s", "refined": "o", "blink": "D", "gemma4": "^",
          "qwen3_vl": "v", "grounded_gemma_linking": "P"}
KEY_CONFIGS = ["population_wikidata", "refined", "blink", "gemma4", "qwen3_vl",
               "grounded_gemma_linking"]
CONDITIONS = ["EXPLICIT", "NOISE_EXPLICIT", "B", "NOISE_B", "A", "NOISE_A"]
CONDITION_LABEL = {"EXPLICIT": "Named", "NOISE_EXPLICIT": "Named\n+ noise", "B": "Ambiguous",
                   "NOISE_B": "Ambiguous\n+ noise", "A": "Clue only", "NOISE_A": "Clue only\n+ noise"}

INK, MUTED, GRID = "#222222", "#666666", "#e3e3e3"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


# ---------------------------------------------------------------- data

def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    """Shares are written as percentages with one decimal; counts stay whole."""
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([f"{v:.1f}" if isinstance(v, float) else v for v in row])


def ordered(names) -> list[str]:
    order = config_order()
    return sorted(names, key=lambda n: row_order(n, order))


def load_scored(split: str) -> dict[str, list[dict]]:
    """{configuration: scored records of the split}, each record joined to its
    dataset row (`_row`) on the post text."""
    dataset = {r["text"]: r for r in read_csv(DATASET_PATH)}
    records = attach_split(load_predictions(find_prediction_files()))
    out = defaultdict(list)
    for rec in records:
        if rec.get("split") != split:
            continue
        scored = score_record(rec)
        scored["_row"] = dataset.get(rec.get("text"), {})
        out[rec["config_name"]].append(scored)
    return out


def pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


ANSWER_KINDS = ["correct", "inside or containing the place", "the clue entity itself",
                "other, right country", "same-name place elsewhere", "other, wrong country",
                "no answer"]
KIND_COLOUR = dict(zip(ANSWER_KINDS, [CORRECT, GUM_LEAF, GRANDIS, INDIAN_YELLOW, CORAL, TERRA_COTTA, NO_ANSWER]))


def answer_kind(rec: dict) -> str:
    """What the returned item was, for a post that is not an exact match."""
    row, pred, gold = rec["_row"], rec.get("predicted_qid"), rec.get("place_qid")
    if rec["exact"]:
        return "correct"
    if not pred:
        return "no answer"
    if pred in [q for q in re.split(r"[|;,\s]+", row.get("clue_qids") or "") if q]:
        return "the clue entity itself"
    if row.get("rival_qid") and pred == row["rival_qid"]:
        return "same-name place elsewhere"
    if (containment() or {}).get((pred, gold)) in ("inside", "contains"):
        return "inside or containing the place"
    return "other, right country" if rec["country_correct"] else "other, wrong country"


# ---------------------------------------------------------------- figures

def save(fig, out: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {out / name}.png / .pdf")


def bar_axes(ax, labels: list[str], xlabel: str | None, xmax: float = 100) -> None:
    """The common look of every horizontal bar chart."""
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.set_xlim(0, xmax)
    ax.spines["left"].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)


def segment(ax, y: float, left: float, value: float, colour: str, min_label: float) -> None:
    """One piece of a stacked bar, with its value when it is wide enough."""
    ax.barh(y, value, left=left, height=BAR_HEIGHT, color=colour, edgecolor="white", linewidth=1.2)
    if value >= min_label:
        ax.text(left + value / 2, y, f"{value:.0f}", ha="center", va="center", fontsize=7.5,
                color="white" if colour in DARK_FILLS else INK)


def bottom_legend(fig, handles, labels, ncol: int, y: float) -> None:
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y))


def swatches(colours) -> list:
    return [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours]


def fig_overall(split: str, out: Path) -> None:
    rows = {r["config"]: r for r in read_csv(SUMMARY_DIR / f"table_bootstrap_{split}.csv")}
    names = ordered(rows)[::-1]
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    for i, name in enumerate(names):
        v, lo, hi = (100 * float(rows[name][k]) for k in ("exact", "lo", "hi"))
        ax.barh(i, v, height=BAR_HEIGHT, color=COLOUR[name], edgecolor="white", linewidth=1.2)
        ax.plot([lo, hi], [i, i], color=INK, linewidth=1, solid_capstyle="butt")
        ax.text(max(hi, v) + 1.2, i, f"{v:.1f}", va="center", fontsize=8)
    bar_axes(ax, [LABEL[n] for n in names], "Exact match, %", xmax=75)
    save(fig, out, "fig1_overall")


def fig_heatmap(scored: dict, out: Path) -> None:
    names = ordered(scored)
    grid = [[pct(sum(r["exact"] for r in scored[n] if r["condition"] == c),
                 sum(1 for r in scored[n] if r["condition"] == c)) for c in CONDITIONS]
            for n in names]
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    im = ax.imshow(grid, cmap=HEATMAP, vmin=0, vmax=100, aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 70 else INK)
    ax.set_xticks(range(len(CONDITIONS)), [CONDITION_LABEL[c] for c in CONDITIONS])
    ax.set_yticks(range(len(names)), [LABEL[n] for n in names])
    ax.tick_params(length=0, colors=INK)
    ax.xaxis.tick_top()
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(CONDITIONS))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(names))], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    bar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("Exact match, %")
    bar.outline.set_visible(False)
    save(fig, out, "fig2_condition_heatmap")


def fig_noise(scored: dict, out: Path) -> None:
    pairs = [("EXPLICIT", "NOISE_EXPLICIT", "Named place"), ("B", "NOISE_B", "Ambiguous name"),
             ("A", "NOISE_A", "Clue only")]
    steps = ["clean", "demonym", "entity"]
    table = []
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.3), sharey=True)
    for ax, (clean, noisy, title) in zip(axes, pairs):
        for name in KEY_CONFIGS:
            groups = {"clean": [r for r in scored[name] if r["condition"] == clean]}
            for kind in ("demonym", "entity"):
                groups[kind] = [r for r in scored[name]
                                if r["condition"] == noisy and r.get("noise_kind") == kind]
            ys = [pct(sum(r["exact"] for r in groups[s]), len(groups[s])) for s in steps]
            for s, y in zip(steps, ys):
                table.append([name, clean, s, len(groups[s]), y])
            ax.plot(range(3), ys, color=COLOUR[name], linewidth=2, marker=MARKER[name],
                    markersize=6, markeredgecolor="white", markeredgewidth=1, label=LABEL[name])
        ax.set_title(title, loc="left")
        ax.set_xticks(range(3), ["none", "demonym", "entity"])
        ax.set_xlabel("Distractor")
        ax.set_xlim(-0.25, 2.25)
        ax.set_ylim(0, 100)
        ax.grid(axis="y")
        ax.tick_params(axis="x", length=0, colors=INK)
    axes[0].set_ylabel("Exact match, %")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    bottom_legend(fig, handles, labels, 3, -0.12)
    save(fig, out, "fig3_noise")
    write_csv(out / "data_fig3_noise.csv", ["config", "condition", "distractor", "n", "exact"], table)


def fig_layers(split: str, out: Path) -> None:
    layers = [("correct", "Correct", CORRECT),
              ("no_mention", "No usable mention or link", GRANDIS),
              ("gold_not_in_candidates", "Correct place not among candidates", INDIAN_YELLOW),
              ("wrong_selection", "Correct place available, another chosen", CORAL),
              ("wrong", "Wrong (no intermediate stage)", TERRA_COTTA),
              ("abstained", "Declined to answer", NO_ANSWER)]
    share = defaultdict(dict)
    for r in read_csv(SUMMARY_DIR / f"table_error_layers_{split}.csv"):
        if r["condition"] == "ALL":
            share[r["config"]][r["layer"]] = 100 * float(r["share_layer"])
    names = ordered(share)[::-1]
    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    for i, name in enumerate(names):
        left = 0.0
        for key, _, colour in layers:
            v = share[name].get(key, 0.0)
            if v:
                segment(ax, i, left, v, colour, 6)
                left += v
    bar_axes(ax, [LABEL[n] for n in names], "Test posts, %")
    bottom_legend(fig, swatches(c for _, _, c in layers), [label for _, label, _ in layers], 2, -0.16)
    save(fig, out, "fig4_error_layers")


def fig_wrong_answers(scored: dict, out: Path) -> None:
    names = ["refined", "blink", "qwen3_vl", "grounded_gemma_linking"]
    table = []
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.4), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, names):
        for i, cond in enumerate(CONDITIONS[::-1]):
            recs = [r for r in scored[name] if r["condition"] == cond]
            counts = Counter(answer_kind(r) for r in recs)
            left = 0.0
            for kind in ANSWER_KINDS:
                v = pct(counts[kind], len(recs))
                table.append([name, cond, kind, counts[kind], len(recs), v])
                if v:
                    segment(ax, i, left, v, KIND_COLOUR[kind], 9)
                    left += v
        ax.set_title(LABEL[name], loc="left")
        bar_axes(ax, [CONDITION_LABEL[c].replace("\n", " ") for c in CONDITIONS[::-1]], None)
    for ax in axes[1]:
        ax.set_xlabel("Test posts in the condition, %")
    fig.tight_layout()
    bottom_legend(fig, swatches(KIND_COLOUR[k] for k in ANSWER_KINDS),
                  [k[0].upper() + k[1:] for k in ANSWER_KINDS], 2, -0.13)
    save(fig, out, "fig5_wrong_answers")
    write_csv(out / "data_fig5_wrong_answers.csv",
              ["config", "condition", "answer", "n", "of", "share"], table)


def fig_prominence(split: str, out: Path) -> None:
    rows = read_csv(SUMMARY_DIR / f"table_bins_{split}.csv")
    panels = [("place_prominence", "Prominence of the place", "Wikipedia language editions",
               [("low_under_30", "under 30"), ("mid_30_to_99", "30 to 99"),
                ("high_100_plus", "100 or more")]),
              ("rival_prominence", "Place against same-name rival", "Which of the two is more prominent",
               [("rival_leads_2x", "rival, 2x"), ("close", "close"),
                ("place_leads_2x", "place, 2x")])]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.4), sharey=True)
    for ax, (bin_name, title, xlabel, levels) in zip(axes, panels):
        for name in KEY_CONFIGS:
            cell = {r["level"]: r for r in rows if r["bin"] == bin_name and r["config"] == name}
            ys = [100 * float(cell[k]["exact"]) for k, _ in levels]
            lo = [100 * (float(cell[k]["exact"]) - float(cell[k]["exact_lo"])) for k, _ in levels]
            hi = [100 * (float(cell[k]["exact_hi"]) - float(cell[k]["exact"])) for k, _ in levels]
            ax.errorbar(range(len(levels)), ys, yerr=[lo, hi], color=COLOUR[name], linewidth=2,
                        elinewidth=0.8, marker=MARKER[name], markersize=6,
                        markeredgecolor="white", markeredgewidth=1, label=LABEL[name])
        ax.set_title(title, loc="left")
        ax.set_xticks(range(len(levels)), [label for _, label in levels])
        ax.set_xlabel(xlabel)
        ax.set_xlim(-0.25, len(levels) - 1 + 0.25)
        ax.set_ylim(0, 100)
        ax.grid(axis="y")
        ax.tick_params(axis="x", length=0, colors=INK)
    axes[0].set_ylabel("Exact match, %")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    bottom_legend(fig, handles, labels, 3, -0.12)
    save(fig, out, "fig6_prominence")


def fig_answer_precision(scored: dict, out: Path) -> None:
    names = [n for n in ordered(scored) if n != "majority_class"]
    table = []
    fig, ax = plt.subplots(figsize=(6.3, 4.3))
    for level in (10, 20, 30, 40, 50, 60):          # lines of equal exact match
        xs = [x / 2 for x in range(2 * level, 201)]
        ax.plot(xs, [100 * level / x for x in xs], color=GRID, linewidth=0.8, zorder=1)
        ax.text(102.6, level, f"{level}", fontsize=7, color=MUTED, va="center")
    ax.text(102.6, 76, "exact\nmatch, %", fontsize=7, color=MUTED, va="center")
    nudge = {"refined": (-1.4, 1.6, "right"), "blink": (0, 3.2, "center"),
             "grounded_gemma_linking": (-1.4, 2.4, "right"), "population_wikidata": (-1.4, -2.6, "right"),
             "gemma4": (-1.4, -1.8, "right"), "qwen3_vl": (-2.0, 3.2, "center"),
             "retrieve_rerank": (-1.4, 0, "right"), "dense_retrieval": (-1.4, 0, "right")}
    for name in names:
        recs = scored[name]
        answered = [r for r in recs if r.get("predicted_qid")]
        rate = pct(len(answered), len(recs))
        precision = pct(sum(r["exact"] for r in answered), len(answered))
        table.append([name, len(recs), rate, precision, pct(sum(r["exact"] for r in recs), len(recs))])
        ax.scatter(rate, precision, s=62, color=COLOUR[name], edgecolor="white", linewidth=1.2, zorder=3)
        dx, dy, ha = nudge.get(name, (1.4, 0, "left"))
        ax.text(rate + dx, precision + dy, LABEL[name], fontsize=8, va="center", ha=ha, zorder=4)
    ax.set_xlim(20, 102)
    ax.set_ylim(0, 80)
    ax.set_xlabel("Test posts answered, %")
    ax.set_ylabel("Exact match among answered posts, %")
    save(fig, out, "fig7_answer_precision")
    write_csv(out / "data_fig7_answer_precision.csv",
              ["config", "n", "answered", "precision_when_answered", "exact"], table)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scored = load_scored(args.split)
    fig_overall(args.split, out)
    fig_heatmap(scored, out)
    fig_noise(scored, out)
    fig_layers(args.split, out)
    fig_wrong_answers(scored, out)
    fig_prominence(args.split, out)
    fig_answer_precision(scored, out)


if __name__ == "__main__":
    main()
