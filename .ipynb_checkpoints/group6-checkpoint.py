# ============================================================
# UK House Index Analysis - Group 6
# ============================================================
# ============================================================
# Libraries
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy import stats
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

hpi_months = np.array(
    [f"{d[6:10]}-{d[3:5]}" for d in hpi["Date"]],
    dtype = "datetime64[M]"
)
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

rate_months = rates["Date"].astype("datetime64[M]")


# --- overview ---
#
#print("> overview rates")
#print("Rows         :", rates.shape[0])
#print("Date range   :", rates["Date"].min(), "to", rates["Date"].max())

#print("> data")
#print(rates[:5])

#> overview rates
#Rows         : 260
#Date range   : 2005-01-31 to 2026-08-31
#> data
#[('2005-01-31', 4.99, 5.2 , '2005-01')
# ('2005-02-28', 5.07, 5.25, '2005-02')
# ('2005-03-31', 5.2 , 5.28, '2005-03')
# ('2005-04-30', 5.16, 5.23, '2005-04')
# ('2005-05-31', 4.91, 5.05, '2005-05')]




# ============================================================
# Function and Definition
# ============================================================
# ============================================================
# Commonly used functions..
# ============================================================

def ask_option(question, choices):
    print(question)
    for i, choice in enumerate(choices, start = 1):
        print(f"  {i}.  {choice}")
    while True:
        answer = input("Input your choice: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        print("Invalid choice, please try again...")

def ask_year_month(question):
    while True:
        answer_year_month = input(question).strip()
        if not(
            len(answer_year_month) == 7 and answer_year_month[4] == "-"
            and answer_year_month[:4].isdigit() and answer_year_month[5:].isdigit()):
            print("Use the format YYYY-MM, e.g. 2015-01.")
            
        elif int(answer_year_month[:4]) < 2005:
            print("The data starts in 2005-01. Please enter 2005-01 or later.")
        elif not 1 <= int(answer_year_month[5:]) <= 12:
            print("The month must be between 01 and 12.")
        else:
            return np.datetime64(answer_year_month, "M")

def build_series(country, rate_name, start, end):
    in_country = hpi["RegionName"] == country
    months = hpi_months[in_country]
    growth = hpi["12mChange"][in_country]

    # Match house price growth and mortgage rate on the month
    common, idx_hpi, idx_rate = np.intersect1d(
        months, rate_months, return_indices = True
    )
    growth = growth[idx_hpi]
    rate = rates[rate_name][idx_rate]

    # Keep the chosen period and drop missing values
    keep = (
        (common >= start) & (common <= end)
        & ~np.isnan(growth) & ~np.isnan(rate)
    )
    return common[keep], growth[keep], rate[keep]

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
# ============================================================
# RQ1: Are higher mortgage interest rates associated with lower annual house price growth?
# ============================================================

def run_q1():

    # Introduction

    print("\n" + "=" * 65)
    print("Q1 | Mortgage Interest Rates and House Price Growth Relationship")
    print("=" * 65)

    print("> Research Question:")
    print("Are higher mortgage interest rates associated with lower")
    print("annual house price growth?")


    print("\n> About this analysis")
    print("This program examines the relationship between mortgage")
    print("interest rates and annual house price growth in the UK.")
    print("It uses descriptive statistics and Pearson correlation to analyze the relationship.")

    print("\n> You will be asked to select:")
    print("1. A UK Region (England, Wales, northern Ireland, Scotland, United Kingdom")
    print("2. Choose between 2-year and 5-year 75% LTV fixed mortgage interest rates")
    print("3. A start and end month for the analysis")

    print("\n" + "-" * 65)
    print("1. Choose a UK country")
    print("" + "-" * 65)
    
    # Choose Region
    countries = ["England", "Scotland", "Wales", "Northern Ireland", "United Kingdom"]
    country_answer = ask_option("Choose the region you want to analyse:", countries)

    print("\n" + "-" * 65)
    print("2. Choose a mortgage rate")
    print("" + "-" * 65)
    print("Rate2y = rate fixed for 2 years. Rate5y = rate fixed for 5 years.")
    print("Both are for a 75% LTV mortgage.")
    # Choose Mortgage Rate (2y or 5y) to analyze.
    rates = ["rate2y", "rate5y"]
    rate_answer = ask_option("Choose term of the mortgage rate:", ["Rate2y", "Rate5y"])

    # Ask Start-Month
    print("\n" + "-" * 65)
    print("3. Choose the start month and end month")
    print("" + "-" * 65)

    print("Type each month as YYYY-MM, for example 2015-06 for June 2015.")
    print("- The start month must be 2005-01 or later.")
    print("- The end month must not be earlier than the start month.")
    print("- The period must cover at least 12 months, because a")
    print("  correlation based on very few months is not reliable.")


    while True:
        start_month = ask_year_month("Determine the start month for analysis: ")
        end_month = ask_year_month("Determine the end month for analysis: ")

        # count the months in the period, including both ends
        n_months = int((end_month - start_month).astype(int)) + 1
        MIN_MONTHS = 12
        
        if n_months < 1:
            print("The end month must not be earlier than the start month.")
        elif n_months < MIN_MONTHS:
            print("That period covers only %d month(s). Please choose at"
                  % n_months)
            print("least %d months." % MIN_MONTHS)
        else:
            break
    
    print("\n" + "-" * 65)
    print(f"Mortgage Interest ({rate_answer}) Relationship with Annual Price Growth in {country_answer} from {start_month} and {end_month}.")
    print("" + "-" * 65)
    

    months, growth, rate = build_series(country_answer, rate_answer, start_month, end_month)

    print("Country      :", country_answer)
    print("Mortgage rate:", rate_answer)
    print("Months used  :", months.shape[0])
    print("Date range   :", months.min(), "to", months.max())

    print("\n--- Descriptive statistics ---")
    # Mortgage Interest Rate
    print("Mortgage Interest Rate:", rate_answer)
    print("Mean:", round(statistics.mean(rate), 2))
    print("Median:", round(statistics.median(rate), 2))
    print("Standard Deviation:", round(statistics.stdev(rate), 2))
    
    # House Price Growth
    print("\nHouse Price Growth (12mChange)")
    print("Mean:", round(statistics.mean(growth), 2))
    print("Median:", round(statistics.median(growth), 2))
    print("Standard Deviation:", round(statistics.stdev(growth), 2))

    pearson = stats.pearsonr(rate, growth)
    fit = stats.linregress(rate, growth)

    print("\n--- Pearson Correlation ---")
    print(f"Pearson r    : {pearson.statistic:.3f} (p = {pearson.pvalue:.4f})")
    print(f"r = {pearson.statistic:.3f}")
    print(f"p-value = {pearson.pvalue:.4f}")
    print("\n")
    if pearson.pvalue >= 0.05:
        print("Result: No statistifically significant linear relationship")
        print("Hence, there is not enough statistical evidence to conclude that mortgage interest rates are linearly related to house price growth.")
    elif pearson.statistic < 0:
        print("Result: Significant negative linear relationship.")
        print("Higher mortgage interest rates tend to be associated with lower annual house price growth.")
        print("Note - this relationship is statistifically siginificant at the 5% level.")
    else:
        print("Result: Significant positive linear relationship.")
        print("Higher mortgage interest rate tend to be associated with higher annual house price growth")
        print("Note - this relationship is statistifically siginificant at the 5% level.")

    plt.scatter(rate, growth)

    line_x = np.linspace(rate.min(), rate.max(), 100)
    line_y = fit.intercept + fit.slope * line_x
    
    plt.plot(line_x, line_y, color="red")
    
    plt.xlabel("Mortgage interest rate (%)")
    plt.ylabel("Annual house price growth (%)")
    plt.title("Mortgage Rates vs House Price Growth")
    
    plt.show()
                




    

# ============================================================
# RQ2
# ============================================================

def run_q2():
        # Introduction
    print("\n" + "=" * 65)
    print("Q2 | New-Build and Existing Home Price Comparison")
    print("=" * 65)

    print("> Research Question:")
    print("How do price gaps and new-build premiums differ between")
    print("two UK areas over a period chosen by the user?")

    print("\n> About this analysis")
    print("This analysis uses monthly data from the UK House Price Index")
    print("to compare new-build and existing home prices in two areas.")
    print("It calculates the average monthly price gap in pounds and")
    print("the average monthly new-build premium as a percentage.")

    print("\n> You will be asked to select:")
    print("1. Two different UK areas")
    print("   - a country, e.g. England, Scotland, Wales, Northern Ireland")
    print("   - a region, e.g. London, North East, South West")
    print("   - a local area, e.g. Aberdeenshire, City of Edinburgh, Manchester")
    print("   You can type part of a name (e.g. 'edin'). If more than one")
    print("   area matches, you will be shown the list and asked again.")
    print("2. A start year and an end year for the analysis")
    
    # Build the four columns Q2 needs from the shared hpi data
    years = hpi["Year-Month"].astype("datetime64[Y]").astype(int) + 1970
    new_text = np.where(np.isfinite(hpi["NewPrice"]),
                        hpi["NewPrice"].astype(str), "")
    old_text = np.where(np.isfinite(hpi["OldPrice"]),
                        hpi["OldPrice"].astype(str), "")

    data = np.column_stack((years.astype(str), hpi["RegionName"],
                            new_text, old_text))

    # Get the user's choices
    region_names = []
    for name in data[:, 1]:
        if name not in region_names:
            region_names.append(name)
    print("\n" + "-" * 65)
    print("1. Choose two different UK areas.")
    print("" + "-" * 65)
    region_1 = choose_region("Enter the first region: ", region_names)
    region_2 = choose_region("Enter the second region: ", region_names)

    max_year = int(str(hpi["Year-Month"].max())[:4])
    print("\n" + "-" * 65)
    print("2. Choose a start year and an end year for the analysis")
    print("" + "-" * 65)
    print("The data is monthly. Both years are included in full, so")
    print("2005 to 2006 means January 2005 to December 2006 (24 months).")

    start_text = input(f"\nEnter the start year (2005 to {max_year}): ").strip()
    end_text = input(f"Enter the end year (2005 to {max_year}): ").strip()

    if not start_text.isdigit() or not end_text.isdigit():
        print("Please enter whole numbers for the years.")
        return

    start_year = int(start_text)
    end_year = int(end_text)

    if start_year < 2005 or end_year > max_year:
        print(f"Please choose years between 2005 and {max_year}.")
        return

    if start_year > end_year:
        print("The start year must not be later than the end year.")
        return

    if region_1.lower() == region_2.lower():
        print("Please choose two different regions.")
        return

    print("\n" + "-" * 65)
    print(f"Price Gap and New-build premiums Analysis between {region_1} and {region_2}.")
    print("" + "-" * 65)
    print("")
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
        # Compare only when both regions have complete monthly data
        expected_months = (end_year - start_year + 1) * 12

        print()
        print("Region:", region)
        print("Period:", start_year, "to", end_year)
        print("Months with both prices: %d of %d" % (count, expected_months))


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

    

    if valid_counts[0] > 0 and valid_counts[1] > 0:
        if (valid_counts[0] != expected_months and valid_counts[1] != expected_months):
            print()
            print("Note: some months are missing from the selected period,")
            print("so the averages use only the months with data.")
            
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

        print()
        print("CONCLUSION")

        gap_1 = average_gaps[0]
        gap_2 = average_gaps[1]
        premium_1 = average_premiums[0]
        premium_2 = average_premiums[1]

        # First area
        if premium_1 >= 0:
            print("- In %s, new-build homes cost MORE than existing homes:"
                  % region_1)
            print("  GBP %.2f more on average, a premium of %.2f%%."
                  % (gap_1, premium_1))
        else:
            print("- In %s, new-build homes cost LESS than existing homes:"
                  % region_1)
            print("  GBP %.2f less on average, a discount of %.2f%%."
                  % (abs(gap_1), abs(premium_1)))

        # Second area
        if premium_2 >= 0:
            print("- In %s, new-build homes cost MORE than existing homes:"
                  % region_2)
            print("  GBP %.2f more on average, a premium of %.2f%%."
                  % (gap_2, premium_2))
        else:
            print("- In %s, new-build homes cost LESS than existing homes:"
                  % region_2)
            print("  GBP %.2f less on average, a discount of %.2f%%."
                  % (abs(gap_2), abs(premium_2)))

        # Both areas together
        if premium_1 >= 0 and premium_2 >= 0:
            if premium_1 >= premium_2:
                print("- Both areas have a new-build premium; it is higher in %s."
                      % region_1)
            else:
                print("- Both areas have a new-build premium; it is higher in %s."
                      % region_2)
        elif premium_1 < 0 and premium_2 < 0:
            print("- Both areas have a new-build discount.")
        else:
            print("- The two areas point in opposite directions, so there is")
            print("  no new-build premium that applies to both.")

        




# ============================================================
# RQ3: Which UK regions had the most volatile monthly house price changes over a user-selected period?
# ============================================================

def run_q3():
    #Introduction
    print("\n" + "=" * 65)
    print("Q3 | Monthly House Price Volatility")
    print("=" * 65)

    print("\n> Research Question:")
    print("Which UK regions had the most volatile monthly house price")
    print("changes over a period selected by the user?")

    print("\n> About this analysis")
    print("This analysis uses monthly data from the UK House Price Index")
    print("(HPI) to compare house price changes across local areas.")
    print("It calculates the standard deviation of monthly percentage")
    print("changes to identify areas with higher and lower volatility.")
    print("A higher standard deviation indicates greater price fluctuations.")

    print("\n> You will be asked to select:")
    print("1. England, Scotland, Wales, Northern Ireland or United Kingdom")
    print("2. A start year and an end year for the analysis")

    #Use the HPI dataset already loaded by the group (from 2005 onward).
    #NumPy changes the CSV heading "1m%Change" to "1mChange".
    #Keep the four columns in the same order as the original Q3.
    monthly_change = hpi["1mChange"]
    change_text = np.where(np.isfinite(monthly_change),
                           monthly_change.astype(str), "")
    data = np.column_stack((hpi["Date"], hpi["RegionName"],
                            hpi["AreaCode"], change_text))

    #Ask users to choose an area
    countries = ["England", "Scotland", "Wales", "Northern Ireland", "United Kingdom"]

    selected_country = ask_option(
        "Choose the region you want to analyse:",
        countries
    )

    #AreaCode prefixes for local authority areas
    area_codes = {
        "England": ("E06", "E07", "E08", "E09"),
        "Scotland": ("S12",),
        "Wales": ("W06",),
        "Northern Ireland": ("N09",)
    }

    #Ask users to select a time period
    start_text = input("Enter the start year(2005 onwards): ").strip()
    end_text = input("Enter the end year(2005 onwards): ").strip()

    if not start_text.isdigit() or not end_text.isdigit():
        print("Please enter whole numbers for the years.")
        return

    start_year = int(start_text)
    end_year = int(end_text)

    if start_year > end_year:
        print("The start year must not be later than the end year.")
        return

    print("Selected country:", selected_country)
    print("Select period:", start_year, "to", end_year)

    #Get the AreaCode prefixes for the selected country
    if selected_country == "United Kingdom":
        selected_prefixes = (
            "E06", "E07", "E08", "E09",
            "S12", "W06", "N09"
        )
    else:
        selected_prefixes = area_codes[selected_country]

    #Keep all UK data for the four-country comparison
    uk_data = []

    for i in range(data.shape[0]):
        year = int(data[i, 0][-4:])

        if start_year <= year <= end_year:
            uk_data.append(data[i])
    
    #filter the data by year
    filtered_data = []

    for i in range(data.shape[0]):
        year = int(data[i,0][-4:])
        area_code = data[i, 2]

        if start_year <= year <= end_year:
            if area_code.startswith(selected_prefixes):
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
    if valid_count == 0:
        print("No valid data found for the selected years.")
        return

    #Get the region names
    region_names = []

    for row in valid_data:
        if row[1] not in region_names:
            region_names.append(row[1])

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
    
    #Select local authority areas across the UK
    local_area_names = []

    area_prefixes = ("E06", "E07", "E08", "E09","S12", "W06", "N09")

    for row in valid_data:
        if row[2].startswith(area_prefixes):
            if row[1] not in local_area_names:
                local_area_names.append(row[1])

    local_region_results = []

    for result in region_results:
        if result[0] in local_area_names:
            local_region_results.append(result)

    #Show regions with the highest volatility
    print()
    local_region_results.sort(key=lambda x: x[2], reverse=True)

    print("Regions with the highest price volatility:")

    for i in range(min(5, len(local_region_results))):
        result = local_region_results[i]
        print(f"{i + 1}. {result[0]}")
        print(f"   Average monthly change: {result[1]:.3f}%")
        print(f"   Standard deviation: {result[2]:.3f}")

    #Show regions with the lowest volatility
    local_region_results.sort(key=lambda x: x[2])

    print()
    print("Regions with the lowest price volatility:")
    
    for i in range(min(5, len(local_region_results))):
        result = local_region_results[i]
        print(f"{i + 1}. {result[0]}")
        print(f"   Average monthly change: {result[1]:.3f}%")
        print(f"   Standard deviation: {result[2]:.3f}")

    #Compare the four UK countries
    four_countries = ["England", "Scotland", "Wales", "Northern Ireland"]

    country_results = []

    for country in four_countries:
        change_values = []

        for row in uk_data:
            if row[1] == country and row[3].strip() != "":
                change_values.append(float(row[3]))

        if len(change_values) > 1:
            average_change = statistics.mean(change_values)
            standard_deviation = statistics.stdev(change_values)

            country_results.append([
                country,
                average_change,
                standard_deviation
            ])

    if len(country_results) < 4:
        print("Note: Not all four countries have enough data.")

    print()
    print("Price volatility in the four UK countries:")

    for i, result in enumerate(country_results, 1):
        print(f"\n{i}. {result[0]}")
        print(f"   Average monthly change: {result[1]:.3f}%")
        print(f"   Standard deviation: {result[2]:.3f}")

    #Draw the bar chart
    country_names = []
    country_volatility = []

    for result in country_results:
        country_names.append(result[0])
        country_volatility.append(result[2])

    if not country_names:
        print("No country data available to plot.")
        return

    plt.figure(figsize=(8, 5))
    plt.bar(country_names, country_volatility)

    plt.title("House Price Volatility in the UK Countries")
    plt.xlabel("Country")
    plt.ylabel("Standard Deviation")

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


# ============================================================
# RQ4: How have house prices changed across property types in a selected region and period?
# ============================================================
  

def read_price(text):
    """Return a number, or None if the price is missing or invalid."""
    try:
        price = float(text)
    except (TypeError, ValueError):
        return None

    return price if np.isfinite(price) else None

def run_q4():
    print("\n" + "=" * 65)
    print("Q4 | Property Type Comparison")
    print("=" * 65)

    print("> Research Question:")
    print("Which property type had the largest percentage change in")
    print("average price in the selected region and period?")

    print("\n> About this analysis")
    print("This analysis uses UK House Price Index data to compare")
    print("detached houses, semi-detached houses, terraced houses and flats.")
    print("The line chart shows monthly price trends, and the bar chart")
    print("shows the percentage change from the start to the end month.")

    print("\n> You will be asked to select:")
    print("1. A UK region")
    print("2. A start and end month for the analysis (YYYY-MM)")
    print("3. One or more property types to compare")
    print("4. A chart view (relative growth or prices in pounds)")
    print()
    
    # Set the price columns for each property type
    price_columns = {
        "DetachedPrice": "Detached",
        "SemiDetachedPrice": "Semi-detached",
        "TerracedPrice": "Terraced",
        "FlatPrice": "Flat",
    }
    # Ask the user to choose a region
    print("\nQ4 | Property Type Comparison")
    region = input("Region: ").strip() or "Scotland"

    # Read prices for the selected region, grouped by month.
    prices = {}
    selected_region = region.casefold()

    # Use the shared HPI data to get prices for the selected region.
    for row in hpi:
        if row["RegionName"].casefold() != selected_region:
            continue
    
        region = row["RegionName"]
        month = row["Year-Month"]
    
        prices[month] = {}
    
        for column, name in price_columns.items():
            prices[month][name] = read_price(row[column])

    if not prices:
        print("Region not found. Try Scotland, England, or London.")
        return
    
    # Ask for the start and end months
    start_text = input("Start month (YYYY-MM): ").strip()
    end_text = input("End month (YYYY-MM): ").strip()
    
    try:
        start = np.datetime64(start_text or "2020-01", "M")
        end = np.datetime64(end_text or "2025-01", "M")
   
    except ValueError:
        print("Please enter a month in YYYY-MM format, for example 2020-01.")
        return

    if start >= end:
        print("The end month must be after the start month.")
        return

    if start not in prices or end not in prices:
        print(f"{region} has no data for one or both selected months.")
        return
	# Ask which property types to compare
    options = dict(zip(["1", "2", "3", "4"], price_columns.values()))
    print("1. Detached  2. Semi-detached  3. Terraced  4. Flat")

    choice = input("Property types (e.g. 1,4): ").strip()
    numbers = [item.strip() for item in choice.split(",")] if choice else list(options)

    if any(number not in options for number in numbers):
        print("Please select property types using numbers 1 to 4.")
        return
   
    # Calculate the change in pounds and as a percentage
    property_types = [options[number] for number in dict.fromkeys(numbers)]
    view = input("Chart view (1 = relative growth, 2 = price in £): ").strip() or "1"

    if view not in {"1", "2"}:
        print("Please enter 1 or 2 for the chart view.")
        return

    # Use the same result rows for the printed table, charts.
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

    print(f"\n{region} | {start} to {end}")
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

    fig.suptitle(f"{region} | {start} to {end}", fontweight="bold")
    fig.tight_layout()
    plt.show()


# ============================================================
# Main Program
# ============================================================

def start_function():
    print("=" * 65)
    print("         UK HOUSE PRICE INDEX ANALYSIS - GROUP 6")
    print("=" * 65)

    print("\nWelcome to the UK House Price Analysis Program!")
    print("This program allows you to explore housing market trends")
    print("across the United Kingdom using historical house price data")
    print("and mortgage interest rates.")
    print("\nYou can compare regions, examine house price changes,")
    print("and explore relationships between mortgage rates and prices.")

    print("-" * 65)

    questions = [
        "Are higher mortgage interest rates associated with lower annual house price growth?",
        "How do price gaps and new-build premiums differ between two selected areas over a user-selected period?",
        "Which UK regions had the most volatile monthly house price changes over a user-selected period?",
        "Which property types (detached, semi-detached, terraced or flats) experienced the greatest percentage change in house prices ?",
        "Exit"
    ]

    while True:
        choice = ask_option(
            "\nPlease select a Research Question to explore:",
            questions
        )

        if choice == questions[0]:
            run_q1()
        elif choice == questions[1]:
            run_q2()
        elif choice == questions[2]:
            run_q3()
        elif choice == questions[3]:
            run_q4()
        else:
            break

        next_step = ask_option(
            "\nWhat would you like to do next?",
            ["Choose another question", "Exit"]
        )

        if next_step == "Exit":
            break

    print("Goodbye!")


if __name__ == "__main__":
    start_function()