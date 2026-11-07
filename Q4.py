from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def run_q4():
    csv_path = Path("uk_house_index_data_aug2026.csv")

    price_columns = {
        "DetachedPrice": "Detached",
        "SemiDetachedPrice": "Semi-detached",
        "TerracedPrice": "Terraced",
        "FlatPrice": "Flat",
    }

    data = pd.read_csv(csv_path, usecols=["Date", "RegionName", *price_columns])

    print("\nQ4 | Property Type Comparison")
    region = input("Region (Enter for Scotland): ").strip() or "Scotland"
    region_data = data.loc[
        data["RegionName"].str.casefold().eq(region.casefold()),
        ["Date", "RegionName", *price_columns],
    ].copy()

    if region_data.empty:
        print("Region not found. Try Scotland, England, or London.")
        return

    region = region_data["RegionName"].iloc[0]
    region_data["Date"] = pd.to_datetime(
        region_data["Date"], format="%d/%m/%Y"
    )
    prices = (
        region_data.set_index("Date")[list(price_columns)]
        .sort_index()
        .rename(columns=price_columns)
    )

    start_text = input("Start month (YYYY-MM, Enter for 2020-01): ").strip()
    end_text = input("End month (YYYY-MM, Enter for 2025-01): ").strip()

    try:
        start = pd.Period(start_text or "2020-01", freq="M").start_time
        end = pd.Period(end_text or "2025-01", freq="M").start_time
    except ValueError:
        print("Please enter a month in YYYY-MM format, for example 2020-01.")
        return

    if start >= end:
        print("The end month must be after the start month.")
        return

    if start not in prices.index or end not in prices.index:
        print(f"{region} has no data for one or both selected months.")
        return

    options = dict(zip(["1", "2", "3", "4"], price_columns.values()))
    print("1. Detached  2. Semi-detached  3. Terraced  4. Flat")
    choice = input("Property types (e.g. 1,4 or Enter for all): ").strip()
    numbers = [item.strip() for item in choice.split(",")] if choice else list(options)

    if any(number not in options for number in numbers):
        print("Please select property types using numbers 1 to 4.")
        return

    types = [options[number] for number in dict.fromkeys(numbers)]
    view = input("Chart view (1 = relative growth, 2 = price in £): ").strip() or "1"

    if view not in {"1", "2"}:
        print("Please enter 1 or 2 for the chart view.")
        return

    monthly = prices.loc[start:end, types]
    result = pd.DataFrame({
        "Start price (£)": monthly.loc[start],
        "End price (£)": monthly.loc[end],
    })

    valid = result.notna().all(axis=1) & result["Start price (£)"].gt(0)
    if not valid.all():
        print("Missing or invalid endpoint prices:", ", ".join(result.index[~valid]))

    result = result.loc[valid].copy()
    if result.empty:
        print("No selected property type has usable prices for both months.")
        return

    result["Change (£)"] = result["End price (£)"] - result["Start price (£)"]
    result["Change (%)"] = result["Change (£)"] / result["Start price (£)"] * 100
    result = result.sort_values("Change (%)", ascending=False)

    print(f"\n{region} | {start:%b %Y} to {end:%b %Y}")
    print(result.round(2).to_string())

    trend = monthly[result.index]
    if view == "1":
        trend = trend.div(result["Start price (£)"]).mul(100)

    colors = {
        "Detached": "#355C9A",
        "Semi-detached": "#258277",
        "Terraced": "#BC5B20",
        "Flat": "#8064A2",
    }

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    for name in trend.columns:
        left.plot(trend.index, trend[name], label=name, color=colors[name], linewidth=2)

    if view == "1":
        left.axhline(100, color="grey", linestyle="--", linewidth=1)

    left.set_title("Monthly price trends", loc="left")
    left.set_ylabel("Index (start month = 100)" if view == "1" else "Average price (£)")
    left.legend(frameon=False)
    left.grid(axis="y", alpha=0.2)

    changes = result["Change (%)"].sort_values()
    bars = right.barh(changes.index, changes, color=[colors[name] for name in changes.index])
    right.bar_label(bars, labels=[f"{value:+.1f}%" for value in changes], padding=4)
    right.axvline(0, color="grey", linewidth=1)
    right.margins(x=0.25)
    right.set_title("Cumulative price change", loc="left")
    right.set_xlabel("Change over selected period (%)")

    fig.suptitle(f"{region} | {start:%b %Y} to {end:%b %Y}", fontweight="bold")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_q4()# ---
    
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
