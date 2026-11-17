import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    base_path = Path(__file__).resolve().parent
    csv_path = base_path / "shopping_trends DATASET normalization.csv"

    # Read the CSV
    df = pd.read_csv(csv_path)

    customer_col = "Customer ID"
    purchases_col = "Previous Purchases"

    # Aggregate in case customer IDs repeat; take the max previous purchases per customer
    customer_freq = (
        df.groupby(customer_col)[purchases_col]
        .max()
        .reset_index(name="previous_purchases")
    )

    values = customer_freq["previous_purchases"]

    # Create bins for clearer strategic interpretation
    bins = [0, 1, 3, 5, 10, 20, values.max()]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(values, bins=bins, edgecolor="black", color="#4C72B0")

    ax.set_title(
        "Distribution of Customer Purchase History",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Number of Previous Purchases per Customer", fontsize=12)
    ax.set_ylabel("Number of Customers", fontsize=12)

    plt.tight_layout()

    # Save figure in the same folder
    out_path = base_path / "customer_previous_purchases_histogram.png"
    fig.savefig(out_path, dpi=150)

    # Also display if environment supports it
    plt.show()


if __name__ == "__main__":
    main()

