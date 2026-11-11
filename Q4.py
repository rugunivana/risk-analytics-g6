import csv
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def read_price(text):
    """Return a number, or None if the price is missing or invalid."""
    try:
        price = float(text)
    except (TypeError, ValueError):
        return None

    return price if math.isfinite(price) else None


def run_q4():
    csv_path = Path("uk_house_index_data_aug2026.csv")

    price_columns = {
        "DetachedPrice": "Detached",
        "SemiDetachedPrice": "Semi-detached",
        "TerracedPrice": "Terraced",
        "FlatPrice": "Flat",
    }

    print("\nQ4 | Property Type Comparison")
    region = input("Region (Enter for Scotland): ").strip() or "Scotland"

    # Read prices for the selected region, grouped by month.
    prices = {}
    selected_region = region.casefold()

    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["RegionName"].casefold() != selected_region:
                continue

            region = row["RegionName"]
            month = datetime.strptime(row["Date"], "%d/%m/%Y")
            prices[month] = {
                name: read_price(row[column])
                for column, name in price_columns.items()
            }

    if not prices:
        print("Region not found. Try Scotland, England, or London.")
        return

    start_text = input("Start month (YYYY-MM, Enter for 2020-01): ").strip()
    end_text = input("End month (YYYY-MM, Enter for 2025-01): ").strip()

    try:
        start = datetime.strptime(start_text or "2020-01", "%Y-%m")
        end = datetime.strptime(end_text or "2025-01", "%Y-%m")
    except ValueError:
        print("Please enter a month in YYYY-MM format, for example 2020-01.")
        return

    if start >= end:
        print("The end month must be after the start month.")
        return

    if start not in prices or end not in prices:
        print(f"{region} has no data for one or both selected months.")
        return

    options = dict(zip(["1", "2", "3", "4"], price_columns.values()))
    print("1. Detached  2. Semi-detached  3. Terraced  4. Flat")

    choice = input("Property types (e.g. 1,4 or Enter for all): ").strip()
    numbers = [item.strip() for item in choice.split(",")] if choice else list(options)

    if any(number not in options for number in numbers):
        print("Please select property types using numbers 1 to 4.")
        return

    property_types = [options[number] for number in dict.fromkeys(numbers)]
    view = input("Chart view (1 = relative growth, 2 = price in £): ").strip() or "1"

    if view not in {"1", "2"}:
        print("Please enter 1 or 2 for the chart view.")
        return

    # Use the same result rows for the printed table, charts and CSV.
    results = []
    missing_types = []

    for name in property_types:
        start_price = prices[start][name]
        end_price = prices[end][name]

        if start_price is None or end_price is None or start_price <= 0:
            missing_types.append(name)
            continue

        change = end_price - start_price
        change_pct = change / start_price * 100
        results.append((name, start_price, end_price, change, change_pct))

    if missing_types:
        print("Missing or invalid endpoint prices:", ", ".join(missing_types))

    if not results:
        print("No selected property type has usable prices for both months.")
        return

    results.sort(key=lambda row: row[4], reverse=True)

    headers = ("Property type", "Start price (£)", "End price (£)", "Change (£)", "Change (%)")

    print(f"\n{region} | {start:%b %Y} to {end:%b %Y}")
    print(f"{headers[0]:<16} {headers[1]:>16} {headers[2]:>16} "
          f"{headers[3]:>14} {headers[4]:>12}")

    for name, start_price, end_price, change, change_pct in results:
        print(f"{name:<16} {start_price:>16,.2f} {end_price:>16,.2f} "
              f"{change:>14,.2f} {change_pct:>12.2f}")

    colors = {
        "Detached": "#355C9A",
        "Semi-detached": "#258277",
        "Terraced": "#BC5B20",
        "Flat": "#8064A2",
    }

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    months = sorted(month for month in prices if start <= month <= end)

    for name, start_price, *_ in results:
        monthly_prices = [prices[month][name] for month in months]

        if view == "1":
            monthly_prices = [
                price / start_price * 100 if price is not None else None
                for price in monthly_prices
            ]

        left.plot(months, monthly_prices, label=name, color=colors[name], linewidth=2)

    if view == "1":
        left.axhline(100, color="grey", linestyle="--", linewidth=1)

    left.set_title("Monthly price trends", loc="left")
    left.set_ylabel("Index (start month = 100)" if view == "1" else "Average price (£)")
    left.legend(frameon=False)
    left.grid(axis="y", alpha=0.2)

    # Draw in reverse order so the largest change appears at the top.
    chart_rows = list(reversed(results))
    names = [row[0] for row in chart_rows]
    changes = [row[4] for row in chart_rows]

    bars = right.barh(names, changes, color=[colors[name] for name in names])
    right.bar_label(bars, labels=[f"{value:+.1f}%" for value in changes], padding=4)
    right.axvline(0, color="grey", linewidth=1)
    right.margins(x=0.25)
    right.set_title("Cumulative price change", loc="left")
    right.set_xlabel("Change over selected period (%)")

    fig.suptitle(f"{region} | {start:%b %Y} to {end:%b %Y}", fontweight="bold")
    fig.tight_layout()
    plt.show()

    save_choice = input("\nSave the result table as CSV? (yes/no): ").strip().lower()

    if save_choice == "yes":
        output_path = Path("Q4_results.csv").resolve()

        with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            writer.writerows(results)

        print(f"Results saved to: {output_path}")
    elif save_choice == "no":
        print("Results were not saved.")
    else:
        print("Please enter yes or no.")


if __name__ == "__main__":
    run_q4()