import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    categories = ["Single-hop", "Multi-hop", "Long-context"]

    alpha_fact = np.array([0.52, 0.34, 0.20])
    alpha_sentence = np.array([0.30, 0.45, 0.29])
    alpha_chunk = np.array([0.18, 0.21, 0.51])

    # Use STIXGeneral instead of Times New Roman.
    # STIXGeneral is usually bundled with matplotlib and looks close to Times.
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    x = np.arange(len(categories))
    width = 0.24

    fig, ax = plt.subplots(figsize=(4.7, 2.45))

    colors = {
        "Fact": "#4C78A8",
        "Sentence": "#F58518",
        "Chunk": "#54A24B",
    }

    bars_fact = ax.bar(x - width, alpha_fact, width, label="Fact", color=colors["Fact"])
    bars_sentence = ax.bar(x, alpha_sentence, width, label="Sentence", color=colors["Sentence"])
    bars_chunk = ax.bar(x + width, alpha_chunk, width, label="Chunk", color=colors["Chunk"])

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.010,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5
            )

    add_labels(bars_fact)
    add_labels(bars_sentence)
    add_labels(bars_chunk)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Average Preference")
    ax.set_ylim(0, 0.66)
    ax.set_yticks(np.arange(0, 0.61, 0.1))

    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.35)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.1,
        borderaxespad=0.0,
    )

    ax.margins(x=0.03)
    plt.tight_layout(pad=0.1)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(out_dir, "mgqa_alpha_by_source.pdf")
    png_path = os.path.join(out_dir, "mgqa_alpha_by_source.png")

    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    plt.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.015)

    print(f"Saved PDF to: {pdf_path}")
    print(f"Saved PNG to: {png_path}")


if __name__ == "__main__":
    main()