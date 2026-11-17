import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    base_path = Path(__file__).resolve().parent
    csv_path = base_path / "shopping_trends DATASET normalization.csv"

    df = pd.read_csv(csv_path)

    age_col = "Age"
    revenue_col = "Purchase Amount (USD)"

    # Define age demographic bins
    bins = [0, 17, 25, 35, 45, 55, 65, 120]
    labels = [
        "Under 18",
        "18–25",
        "26–35",
        "36–45",
        "46–55",
        "56–65",
        "66+",
    ]

    df["Age Group"] = pd.cut(
        df[age_col],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )

    # Aggregate total revenue per age group
    age_revenue = (
        df.groupby("Age Group")[revenue_col]
        .sum()
        .reset_index(name="Total Revenue")
    )

    # Plot histogram-style bar chart
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        age_revenue["Age Group"].astype(str),
        age_revenue["Total Revenue"],
        color="#4C72B0",
        edgecolor="black",
    )

    ax.set_title(
        "Revenue Contribution by Age Demographic",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Age Group", fontsize=12)
    ax.set_ylabel("Total Revenue (USD)", fontsize=12)

    for i, v in enumerate(age_revenue["Total Revenue"]):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    out_path = base_path / "revenue_by_age_histogram.png"
    fig.savefig(out_path, dpi=150)

    plt.show()


if __name__ == "__main__":
    main()

