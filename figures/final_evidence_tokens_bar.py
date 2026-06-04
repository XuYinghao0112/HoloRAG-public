import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    # =========================================================
    # Final evidence tokens from your experiment sheets
    # =========================================================
    datasets = ["NQ", "MuSiQue", "2Wiki", "HotpotQA", "NarrativeQA"]

    holorag = np.array([253.6, 730.1, 713.9, 753.0, 3158.8])
    hipporag2 = np.array([759.6, 1218.0, 1130.5, 1236.4, 5091.5])
    naiverag = np.array([757.3, 1220.0, 1130.2, 1232.3, 5080.4])
    raptor = np.array([577.3, 1176.5, 1175.8, 1172.7, 2535.9])

    # Ratio relative to HoloRAG
    hipporag2_ratio = hipporag2 / holorag
    naiverag_ratio = naiverag / holorag
    raptor_ratio = raptor / holorag

    # =========================================================
    # Plot settings
    # =========================================================
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    y = np.arange(len(datasets))
    height = 0.22   # thicker bars than before

    fig, ax = plt.subplots(figsize=(5.0, 2.55))  # slightly tighter canvas

    colors = {
        "NaiveRAG": "#4C78A8",   # blue
        "HippoRAG2": "#F58518",  # orange
        "RAPTOR": "#54A24B",     # green
    }

    # Order: NaiveRAG / HippoRAG2 / RAPTOR
    ax.barh(
        y - height,
        naiverag_ratio,
        height,
        label="NaiveRAG",
        color=colors["NaiveRAG"],
        edgecolor="none",
    )
    ax.barh(
        y,
        hipporag2_ratio,
        height,
        label="HippoRAG2",
        color=colors["HippoRAG2"],
        edgecolor="none",
    )
    ax.barh(
        y + height,
        raptor_ratio,
        height,
        label="RAPTOR",
        color=colors["RAPTOR"],
        edgecolor="none",
    )

    # Reference line: HoloRAG = 1.0
    ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.9)

    # =========================================================
    # Axes formatting
    # =========================================================
    ax.set_yticks(y)
    ax.set_yticklabels(datasets)
    ax.invert_yaxis()

    ax.set_xlabel("Token Ratio Relative to HoloRAG (HoloRAG = 1.0)")
    ax.set_xlim(0, 3.15)

    ax.grid(axis="x", linestyle="--", linewidth=0.45, alpha=0.35)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=1.1,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    # reduce extra internal margins
    ax.margins(x=0.01, y=0.08)

    plt.tight_layout(pad=0.15)

    # =========================================================
    # Save figure
    # =========================================================
    out_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(out_dir, "final_evidence_token_ratio.pdf")
    png_path = os.path.join(out_dir, "final_evidence_token_ratio.png")

    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    plt.savefig(png_path, dpi=500, bbox_inches="tight", pad_inches=0.015)

    print(f"Saved PDF to: {pdf_path}")
    print(f"Saved PNG to: {png_path}")

    plt.show()


if __name__ == "__main__":
    main()