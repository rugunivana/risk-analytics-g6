import numpy as np
import statistics
import matplotlib.pyplot as plt

def run_q3():
    # Introduction
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

    # Use the HPI dataset already loaded by the group (from 2005 onward).
    # NumPy changes the CSV heading "1m%Change" to "1mChange".
    # Keep the four columns in the same order as the original Q3.
    monthly_change = hpi["1mChange"]
    change_text = np.where(np.isfinite(monthly_change),
                           monthly_change.astype(str), "")
    data = np.column_stack((hpi["Date"], hpi["RegionName"],
                            hpi["AreaCode"], change_text))

    # Ask users to choose an area
    countries = ["England", "Scotland", "Wales", "Northern Ireland", "United Kingdom"]

    selected_country = ask_option(
        "Choose the region you want to analyse:",
        countries
    )

    # AreaCode prefixes for local authority areas
    area_codes = {
        "England": ("E06", "E07", "E08", "E09"),
        "Scotland": ("S12",),
        "Wales": ("W06",),
        "Northern Ireland": ("N09",)
    }

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

    # Compare the four UK countries
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

    for result in country_results:
        print(result[0],
              "Average change:", round(result[1], 3),
              "Standard deviation:", round(result[2], 3)
              )

    # Draw the bar chart
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
