import numpy as np
import statistics
import matplotlib.pyplot as plt


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

if __name__ == "__main__":
    run_q3()