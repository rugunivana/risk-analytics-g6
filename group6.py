# ============================================================
# UK House Index Analysis - Group 6
# ============================================================
# ============================================================
# Libraries
# ============================================================

import numpy as np
import scipy
import matplotlib.pyplot as plt
import statistics

# ============================================================
# Function and Definition
# ============================================================
# ============================================================
# RQ1
# ============================================================

# Start here


# ============================================================
# RQ2
# ============================================================
def choose_region(prompt, region_names):
    while True:
        search = input(prompt).strip().lower()

        if search == "":
            print("Please enter a region name.")
            continue

        matches = []

        for name in region_names:
            if search == name.lower():
                return name

            if search in name.lower():
                matches.append(name)

        if len(matches) == 1:
            print("Matched region:", matches[0])
            return matches[0]

        if len(matches) > 1:
            print("Multiple regions match. Please enter a more specific name:")
            for name in matches:
                print("-", name)
        else:
            print("No matching region found. Please try again.")

def run_q2():
    file_path = "uk_house_index_data_aug2026.csv"

    # Read the column names
    columns = np.loadtxt(
        fname = file_path,
        delimiter = ",",
        dtype = str,
        max_rows = 1
    )

    for i in range(len(columns)):
        if columns[i] == "Date":
            date_column = i
        if columns[i] == "RegionName":
            region_column = i
        if columns[i] == "NewPrice":
            new_column = i
        if columns[i] == "OldPrice":
            old_column = i

    # Read the required columns
    data = np.loadtxt(
        fname = file_path,
        delimiter = ",",
        dtype = str,
        skiprows = 1,
        usecols = (
            date_column,
            region_column,
            new_column,
            old_column
        )
    )

    # Get the user's choices
    region_names = np.unique(data[:, 1])
    region_1 = choose_region("Enter the first region: ", region_names)
    region_2 = choose_region("Enter the second region: ", region_names)

    start_text = input("Enter the start year: ").strip()
    end_text = input("Enter the end year: ").strip()

    if not start_text.isdigit() or not end_text.isdigit():
        print("Please enter whole numbers for the years.")
        return

    start_year = int(start_text)
    end_year = int(end_text)

    if start_year > end_year:
        print("The start year must not be later than the end year.")
        return

    if region_1.lower() == region_2.lower():
        print("Please choose two different regions.")
        return

    regions = [region_1, region_2]
    average_gaps = []
    average_premiums = []
    valid_counts = []

    # Calculate results for each region
    for region in regions:
        total_gap = 0
        total_premium = 0
        count = 0

        for i in range(data.shape[0]):
            if data[i, 1].lower() == region.lower():
                year = int(data[i, 0][-4:])

                if start_year <= year <= end_year:
                    if data[i, 2] != "" and data[i, 3] != "":
                        new_price = float(data[i, 2])
                        old_price = float(data[i, 3])

                        if new_price > 0 and old_price > 0:
                            gap = new_price - old_price
                            premium = gap / old_price * 100

                            total_gap = total_gap + gap
                            total_premium = total_premium + premium
                            count = count + 1

        valid_counts.append(count)

        print()
        print("Region:", region)
        print("Period:", start_year, "to", end_year)
        print("Valid monthly observations:", count)

        if count > 0:
            average_gap = total_gap / count
            average_premium = total_premium / count

            average_gaps.append(average_gap)
            average_premiums.append(average_premium)

            print("Average monthly price gap: GBP %.2f" % average_gap)
            print("Average monthly new-build premium: %.2f%%"
                  % average_premium)

        else:
            average_gaps.append(0)
            average_premiums.append(0)
            print("No valid data found. Check the region name and years.")

    # Compare only when both regions have complete monthly data
    expected_months = (end_year - start_year + 1) * 12

    if valid_counts[0] > 0 and valid_counts[1] > 0:
        if (valid_counts[0] == expected_months
                and valid_counts[1] == expected_months):
            print()
            print("COMPARISON")

            gap_difference = average_gaps[0] - average_gaps[1]
            premium_difference = average_premiums[0] - average_premiums[1]

            if gap_difference > 0:
                print("The larger average price gap is in:", regions[0])
            elif gap_difference < 0:
                print("The larger average price gap is in:", regions[1])
            else:
                print("Both regions have the same average price gap.")

            print("Difference in average price gaps: GBP %.2f"
                  % abs(gap_difference))

            if premium_difference > 0:
                print("The higher average new-build premium is in:",
                      regions[0])
            elif premium_difference < 0:
                print("The higher average new-build premium is in:",
                      regions[1])
            else:
                print("Both regions have the same average premium.")

            print("Difference in average premiums: %.2f percentage points"
                  % abs(premium_difference))

        else:
            print()
            print("Some months are missing from the selected period.")
            print("Choose a period with complete data for both regions "
                  "to compare them.")



# ============================================================
# RQ3
# ============================================================

def run_q3():
    file_path = "uk_house_index_data_aug2026.csv"

    #Read the column names
    columns = np.loadtxt(
        fname=file_path,
        delimiter=",",
        dtype=str,
        max_rows=1
    )

    for i in range(len(columns)):
        if columns[i] == "Date":
            date_column = i
        if columns[i] == "RegionName":
            region_column = i
        if columns[i] == "AreaCode":
            area_column = i
        if columns[i] == "1m%Change":
            change_column = i

    #Read the required data
    data = np.loadtxt(
        fname=file_path,
        delimiter=",",
        dtype=str,
        skiprows=1,
        usecols=(
            date_column,
            region_column,
            area_column,
            change_column
        )
    )

    #Ask users to select a time period
    start_text = input("Enter the start year: ").strip()
    end_text = input("Enter the end year: ").strip()

    if not start_text.isdigit() or not end_text.isdigit():
        print("Please enter whole numbers for the years.")
        return

    start_year = int(start_text)
    end_year = int(end_text)

    if start_year > end_year:
        print("The start year must not be later than the end year.")
        return

    print("Select period:", start_year, "to", end_year)

    #filter the data by year
    filtered_data = []

    for i in range(data.shape[0]):
        year = int(data[i,0][-4:])

        if start_year <= year <= end_year:
            filtered_data.append(data[i])

    #check for missing values
    valid_data = []
    valid_count = 0
    missing_count = 0

    for row in filtered_data:
        change_text = row[3].strip()

        if change_text == "":
            missing_count = missing_count + 1
        else:
            valid_data.append(row)
            valid_count = valid_count +1
    
    print("Number of records:", len(filtered_data))
    print("Valid monthly changes:", valid_count)
    print("Missing monthly changes:", missing_count)

    #Get the region names
    region_names = []

    for row in valid_data:
        if row[1] not in region_names:
            region_names.append(row[1])

    countries = ["England", "Scotland", "Wales", "Northern Ireland"]

    print()
    print("UK countries in the data:")

    for country in countries:
        if country in region_names:
            print(country)

    print()
    print("Number of regions:", len(region_names))

    #calculate average change and standard deviation
    region_results = []

    for region in region_names:
        change_values = []

        for row in valid_data:
            if row[1] == region:
                change_value = float(row[3])
                change_values.append(change_value)

        if len(change_values) > 1:
            average_change = statistics.mean(change_values)
            standard_deviation = statistics.stdev(change_values)
            
            region_results.append([
                str(region),
                average_change,
                standard_deviation
            ])

    local_region_results = []

    for result in region_results:
        if result[0] not in countries:
            local_region_results.append(result)

    #Show regions with the highest volatility
    print()
    local_region_results.sort(key=lambda x: x[2], reverse=True)

    print("Regions with the highest price volatility:")

    for i in range(min(5, len(local_region_results))):
        print(local_region_results[i])

    #Show regions with the lowest volatility
    local_region_results.sort(key=lambda x: x[2])

    print()
    print("Regions with the lowest price volatility:")
    
    for i in range(min(5, len(local_region_results))):
        print(local_region_results[i])

    #compare four UK countries
    print()
    print("Price volatility in the four UK countries:")

    for result in region_results:
        if result[0] in countries:
            print(result[0],
                 "Average change:", round(result[1],3),
                 "Standard deviation:", round(result[2],3)
                 )

    #draw the bar chart
    country_names = []
    country_volatility = []

    for country in countries:
        for result in region_results:
            if result[0] == country:
                country_names.append(country)
                country_volatility.append(result[2])

    plt.figure(figsize=(8,5))
    plt.bar(country_names,country_volatility)

    plt.title("House Price Volatility in the UK Countries")
    plt.xlabel("Country")
    plt.ylabel("Standard Deviation")

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()



# ============================================================
# RQ4
# ============================================================
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

if __name__ == "__main__":
    run_q4()





# ============================================================
# Main Program
# ============================================================

def start_function():
    print("\nUK House Price Index Analysis - Group 6")
    print("1. Within a region, are monthly changes in average house prices "
              "and sales volumes correlated, and do volume changes lead price changes?")
    print("2. How do price gaps and new-build premiums differ between "
              "two selected areas over a user-selected period?")
    print("3. Which UK regions had the most volatile monthly house price "
              "changes over a user-selected period?")
    print("4. Within a region, which property types were most affected "
              "(detached, semi-detached, terraced or flats)?")
    print("0. Exit")

    choice = input("Enter your choice: ").strip()

    if choice == "2":
        run_q2()
    if choice == "3":
        run_q3()
    if choice == "4":
        run_q4()
    else:
        print("Invalid choice. Please enter 2.")


if __name__ == "__main__":
    start_function()
