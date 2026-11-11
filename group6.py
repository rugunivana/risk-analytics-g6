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


if __name__ == "__main__":
    run_q2()

# ============================================================
# RQ3
# ============================================================

# Start here

# ============================================================
# RQ4
# ============================================================

# Start here

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
    else:
        print("Invalid choice. Please enter 2.")


if __name__ == "__main__":
    start_function()
