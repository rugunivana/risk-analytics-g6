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
# Import Data and Filter Data
# ============================================================
# -------------------------
# File Paths
# -------------------------

# house price index data
path_uk_hpi = "uk_house_index_data_aug2026.csv"
# interest rate
path_rate2y75ltv = "IUMBV34.csv"
path_rate5y75ltv = "IUMBV42.csv"

# -------------------------
# UK House Price Index Data
# -------------------------

## Data
hpi = np.genfromtxt(
    fname = path_uk_hpi,
    delimiter = ",",
    dtype = None,
    names = True,
    encoding = "utf-8"
)
## Column
columns = np.genfromtxt(
        fname = path_uk_hpi,
        delimiter = ",",
        dtype = str,
        max_rows = 1
    )
# print(columns)

# add a column called = 'Year-Month' to the existing data; create a new array (hpi_new),.
new_columns_list = hpi.dtype.descr + [("Year-Month", "datetime64[M]")]
hpi_new = np.empty(hpi.shape, dtype = new_columns_list)

# copy all original columns
for col in hpi.dtype.names:
    hpi_new[col] = hpi[col]

# convert Date into Year-Month
for i, d in enumerate(hpi["Date"]):
    hpi_new["Year-Month"][i] = np.datetime64(
        f"{d[6:10]}-{d[3:5]}", "M"
    )

# replace original array
hpi = hpi_new

# print(hpi.dtype.names)
# print(hpi[:5])

# since data Norther Ireland only starts tracking from January 2005, we will filter the data to only include data from January 2005 onwards
hpi = hpi[hpi["Year-Month"] >= np.datetime64("2005-01")]

'''
The UK House Price Index (UK HPI) captures changes in the value of residential properties.
The UK HPI uses sales data collected on residential housing transactions, whether for cash or with a mortgage. Properties have been included:
in England and Wales since January 1995
in Scotland since January 2004
in Northern Ireland since January 2005
Data is available at a national and regional level, as well as counties, local authorities and London boroughs.
'''

# -------------------------
# Interest Rate Data - Combined the rate2y and rate5y of LTV 75% into one array
# -------------------------
# --- rate2y ---    
rate2y75ltv = np.genfromtxt(
    fname = path_rate2y75ltv,
    delimiter = ",",
    dtype = None,
    skip_header = 1,
    names = ["Date","Rate2y"],
    encoding = "utf-8"
)
# print(rate2y75ltv[:5])
# [('31 Jan 1995', 8.13) ('28 Feb 1995', 8.38) ('31 Mar 1995', 8.08) ('30 Apr 1995', 8.29) ('31 May 1995', 8.22)]

'''
there are two things we need to do to clean the data:
1. convert the rate to float
2. change the date format into yyyy-mm-dd format
'''

# step 1. convert the rate to float
def convert_to_float(data, column):
    for i, d in enumerate(data[column]):
        data[column][i] = float(d)
    return data

rate2y75ltv = convert_to_float(rate2y75ltv, "Rate2y")

# step 2. change the date into yyyy-mm-dd format


month = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12"
}


formatted_dates = []
for d in rate2y75ltv['Date']:
    day = d.split(" ")[0]
    mon = month[d.split(" ")[1]]
    year = d.split(" ")[2]

    formatted_date = f"{year}-{mon}-{day.zfill(2)}"
    formatted_dates.append(formatted_date)

formatted_dates = np.array(formatted_dates, dtype="datetime64[D]")

rate2y_new_dtype = [
    ("Date", "datetime64[D]"),
    ("Rate2y", "f8")
]

rate2y75ltv_new = np.empty(rate2y75ltv.shape, dtype=rate2y_new_dtype)
rate2y75ltv_new["Date"] = formatted_dates
rate2y75ltv_new["Rate2y"] = rate2y75ltv["Rate2y"]
rate2y75ltv = rate2y75ltv_new

# --- rate5y ---

rate5y75ltv = np.genfromtxt(
    fname = path_rate5y75ltv,
    delimiter = ",",
    dtype = None,
    skip_header = 1,
    names = ["Date","Rate5y"],
    encoding = "utf-8"
)
# print(rate5y75ltv[:5])

'''
There are two things we need to do to clean the data:
1. remove the " from the rate5y datasets & convert the rate to float
2. change the date format into 20yy-mm-dd format
'''

# step 1. remove the " from the rate5y datasets & convert the rate to float
for col in rate5y75ltv.dtype.names:
    for i, d in enumerate(rate5y75ltv[col]):
        rate5y75ltv[col][i] = d.strip('"')

rate5y75ltv = convert_to_float(rate5y75ltv, "Rate5y")

# step 2. change the date format into 20yy-mm-dd format


#print(rate5y75ltv[:5])
#[('31 Oct 26', '5.19') ('30 Sep 26', '4.98') ('31 Aug 26', '4.78') ('31 Jul 26', '4.58') ('30 Jun 26', '4.62')]
formatted_dates_5y = []
for d in rate5y75ltv['Date']:
    day = d.split(" ")[0]
    mon = month[d.split(" ")[1]]
    year = d.split(" ")[2]

    formatted_date = f"20{year}-{mon}-{day.zfill(2)}"
    formatted_dates_5y.append(formatted_date)

formatted_dates_5y = np.array(formatted_dates_5y, dtype="datetime64[D]")

rate5y_new_dtype = [
    ("Date", "datetime64[D]"),
    ("Rate5y", "f8")
]

rate5y75ltv_new = np.empty(rate5y75ltv.shape, dtype=rate5y_new_dtype)
rate5y75ltv_new["Date"] = formatted_dates_5y
rate5y75ltv_new["Rate5y"] = rate5y75ltv["Rate5y"]
rate5y75ltv = rate5y75ltv_new

# --- combining both rates ---

common_dates, idx_rate2y, idx_rate5y = np.intersect1d(
    rate2y75ltv["Date"], rate5y75ltv["Date"], return_indices = True
)

rates = np.empty(
    common_dates.shape[0],
    dtype=[
        ("Date", "datetime64[D]"),
        ("Rate2y", "f8"),
        ("Rate5y", "f8"),
        ("Year-Month", "datetime64[M]")
    ]
)

# Assign the values
rates["Date"] = common_dates
rates["Rate2y"] = rate2y75ltv["Rate2y"][idx_rate2y]
rates["Rate5y"] = rate5y75ltv["Rate5y"][idx_rate5y]
rates["Year-Month"] = rates["Date"].astype("datetime64[M]")


# --- overview ---
'''
print("> overview rates")
print("Rows         :", rates.shape[0])
print("Date range   :", rates["Date"].min(), "to", rates["Date"].max())

print("> data")
print(rates[:5])

> overview rates
Rows         : 260
Date range   : 2005-01-31 to 2026-08-31
> data
[('2005-01-31', 4.99, 5.2 , '2005-01')
 ('2005-02-28', 5.07, 5.25, '2005-02')
 ('2005-03-31', 5.2 , 5.28, '2005-03')
 ('2005-04-30', 5.16, 5.23, '2005-04')
 ('2005-05-31', 4.91, 5.05, '2005-05')]

'''


# ============================================================
# Function and Definition
# ============================================================
# ============================================================
# Commonly used functions..
# ============================================================


def ask_option(question, options):
    print(question)


# ============================================================
# RQ1: Are higher mortgage interest rates associated with lower annual house price growth?
# ============================================================

def run_q1():

    # Choose Region
    countries = ["England", "Scotland", "Wales", "Northern Ireland", "United Kingdom"]
    print("Choose the region you want to analyze:")
    for i, country in enumerate(countries, start = 1):
        print(f"{i}. {country}")
    while True:
        country_answer = input("Choose a number: ")
        if country_answer.isdigit() and 1 <= int(country_answer) <= len(countries):
            selected_country = countries[int(country_answer)-1]
            break
        print("Invalid choice, try again.")

    # Choose Mortgage Rate (2y or 5y) to analyze.
    rates = ["rate2y", "rate5y"]
    print("Choose the term rate of the mortgage:")
    for i, rate in enumerate(rates, start = 1):
        print(f"{i}. {rate}")
    while True: 
        rate_answer = input("Choose a number:")
        if rate_answer.isdigit() and 1 <= int(rate_answer) <= len(rates):
            selected_rate = rates[int(rate_answer)-1]
            break
        print("Invalid choice, try again.")

    # Ask Start-Month
    
    while True:
        start_month_answer = input("Determine start month (YYYY-MM) >= January 2005: ").strip()
        if(
            len(start_month_answer) == 7
            and start_month_answer[4] == "-"
            and start_month_answer[:4].isdigit()
            and start_month_answer[5:].isdigit()
            and int(start_month_answer[:4]) >= 2005
        ):
            try:
                selected_start_month = np.datetime64(start_month_answer, "M")
            except ValueError:
                pass

        print("Use the format YYYY-MM, e.g. 2015-01.")
    


    

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

'''
def run_q4():
    csv_path = "uk_house_index_data_aug2026.csv"

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

'''



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
    else:
        print("Invalid choice. Please enter 2.")
    '''
    if choice == "4":
        run_q4()
    '''



if __name__ == "__main__":
    start_function()
