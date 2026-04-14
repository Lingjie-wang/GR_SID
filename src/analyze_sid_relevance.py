import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path

# =========================
# 1. 构造数据
# =========================
data = [
    # dataset, method, layer_0_coverage, layer_0_count_max, Recall@10, NDCG@10
    ["Beauty", "RK-Means", 1.0000, 242, 0.0621, 0.0350],
    ["Beauty", "R-VQ",     1.0000, 186, 0.0688, 0.0398],
    ["Beauty", "RQ-VAE",   0.4531, 360, 0.0572, 0.0336],

    ["Toys",   "RK-Means", 1.0000, 282, 0.0690, 0.0379],
    ["Toys",   "R-VQ",     1.0000, 293, 0.0666, 0.0361],
    ["Toys",   "RQ-VAE",   0.4141, 563, 0.0553, 0.0305],

    ["Sports", "RK-Means", 1.0000, 303, 0.0374, 0.0201],
    ["Sports", "R-VQ",     0.992188, 306, 0.0371, 0.0200],
    ["Sports", "RQ-VAE",   0.488281, 571, 0.0290, 0.0157],
]

df = pd.DataFrame(
    data,
    columns=[
        "dataset", "method", "layer_0_coverage",
        "layer_0_count_max", "Recall@10", "NDCG@10"
    ]
)

# =========================
# 2. 可选：如果你知道每个数据集的 item 数，可做归一化
# Peak Overload = layer_0_count_max / (num_items / 256)
# 你现在不知道的话，先不用这部分，直接画原始 count_max 即可
# =========================
use_normalized_peak = False

# 把这里替换成你真实的数据集 item 数
dataset_num_items = {
    "Beauty": None,
    "Toys": None,
    "Sports": None,
}

if use_normalized_peak:
    def compute_peak_overload(row):
        n_items = dataset_num_items[row["dataset"]]
        if n_items is None:
            raise ValueError(f'{row["dataset"]} 的 item 数还没填，无法归一化。')
        ideal_avg = n_items / 256.0
        return row["layer_0_count_max"] / ideal_avg

    df["layer_0_peak_overload"] = df.apply(compute_peak_overload, axis=1)
    x2_col = "layer_0_peak_overload"
    x2_label = "Layer-0 Peak Overload"
else:
    x2_col = "layer_0_count_max"
    x2_label = "Layer-0 Count Max"

# =========================
# 3. 作图风格设置
# =========================
colors = {
    "RK-Means": "tab:blue",
    "R-VQ": "tab:orange",
    "RQ-VAE": "tab:green",
}

method_markers = {
    "RK-Means": "o",
    "R-VQ": "s",
    "RQ-VAE": "^",
}

# 你可以在这里切换 y 轴性能指标
y_col = "Recall@10"
# y_col = "NDCG@10"

# =========================
# 3.1 手动分隔线模板（可按子图单独调整）
# key 结构：dataset -> left/right -> {"x": 数值或None, "y": 数值或None}
# x: 竖向虚线位置；y: 横向虚线位置
# None 表示使用默认值（x 默认中位数，y 默认不画）
# =========================
manual_sep_lines = {
    "Beauty": {
        "left": {"x": 0.7, "y": 0.060},
        "right": {"x": 275, "y": 0.060},
    },
    "Toys": {
        "left": {"x": 0.7, "y": 0.062},
        "right": {"x": 450, "y": 0.062},
    },
    "Sports": {
        "left": {"x": 0.7, "y": 0.032},
        "right": {"x": 450, "y": 0.032},
    },
}


def draw_sep_lines(ax, sub_df, dataset, panel, x_col_name):
    cfg = manual_sep_lines.get(dataset, {}).get(panel, {})

    x_val = cfg.get("x")
    y_val = cfg.get("y")

    # 默认竖向线：按该子图 x 中位数；横向线默认不画
    if x_val is None:
        x_val = sub_df[x_col_name].median()

    if x_val is not None:
        ax.axvline(
            x=x_val,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            zorder=1,
        )

    if y_val is not None:
        ax.axhline(
            y=y_val,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            zorder=1,
        )

# =========================
# 4. 画 3x2 子图：每行一个数据集
# =========================
datasets = ["Beauty", "Toys", "Sports"]
fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharey=False)

for row_idx, dataset in enumerate(datasets):
    sub_df = df[df["dataset"] == dataset]

    # 左列：Layer-0 Coverage
    ax_left = axes[row_idx, 0]
    for _, row in sub_df.iterrows():
        ax_left.scatter(
            row["layer_0_coverage"],
            row[y_col],
            marker=method_markers[row["method"]],
            color=colors[row["method"]],
            s=120,
            alpha=0.95
        )
        ax_left.annotate(
            row["method"],
            (row["layer_0_coverage"], row[y_col]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )



    # 右列：Layer-0 Count Max 或 Peak Overload
    ax_right = axes[row_idx, 1]
    for _, row in sub_df.iterrows():
        ax_right.scatter(
            row[x2_col],
            row[y_col],
            marker=method_markers[row["method"]],
            color=colors[row["method"]],
            s=120,
            alpha=0.95
        )
        ax_right.annotate(
            row["method"],
            (row[x2_col], row[y_col]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )

    draw_sep_lines(
        ax=ax_left,
        sub_df=sub_df,
        dataset=dataset,
        panel="left",
        x_col_name="layer_0_coverage",
    )
    draw_sep_lines(
        ax=ax_right,
        sub_df=sub_df,
        dataset=dataset,
        panel="right",
        x_col_name=x2_col,
    )

    # 详细小标题
    # ax_left.set_title(f"{dataset}: Layer-0 Coverage vs {y_col}")
    # ax_right.set_title(f"{dataset}: {x2_label} vs {y_col}")

    # 简单小标题
    ax_left.set_title(f"{dataset}", fontweight="bold")
    ax_right.set_title(f"{dataset}", fontweight="bold")

    ax_left.set_xlabel("Layer-0 Coverage", fontweight="bold")
    ax_right.set_xlabel(x2_label, fontweight="bold")

    ax_left.set_ylabel(y_col, fontweight="bold")
    ax_right.set_ylabel(y_col, fontweight="bold")

    ax_left.grid(True, axis="y", linestyle="-", color="0.7", alpha=0.7)
    ax_right.grid(True, axis="y", linestyle="-", color="0.7", alpha=0.7)
    ax_right.invert_xaxis()

# =========================
# 5. 自定义图例
# =========================
from matplotlib.lines import Line2D

method_legend = [
    Line2D([0], [0], marker=method_markers[m], color='w', label=m,
           markerfacecolor=colors[m], markersize=9)
    for m in colors
]

fig.legend(
    handles=method_legend,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 0.995)
)

plt.tight_layout(rect=[0, 0, 1, 0.98])

# Always save figure to outputs so it works in headless environments.
project_root = Path(__file__).resolve().parents[1]
output_dir = project_root / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "sid_relevance_scatter.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {output_path}")

if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    plt.show()
else:
    plt.close(fig)