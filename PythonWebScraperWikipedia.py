import requests
from bs4 import BeautifulSoup
import pandas as pd

#NOTE: these web scrapers have code taht parse through tables. if the url does not have a table to parse through, it will end up failing.

# Send an HTTP request to the webpage
url = 'https://en.wikipedia.org/wiki/Cloud-computing_comparison'  
#response = requests.get(url)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0 Safari/537.36"
}

response = requests.get(url, headers=headers)
#note: the original code from coursera was just response = requests.get(url). that doesnt work because of access issues so i had to add the headers block 
# and pass it into the response


# Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Print the title of the webpage to verify
print("Title: " + soup.title.text)

# Find the table containing the data (selecting the first table by default)
table = soup.find('table')

# Extract table rows
rows = table.find_all('tr')

# Extract headers from the first row (using <th> tags)
headers = [header.text.strip() for header in rows[0].find_all('th')]

# Loop through the rows and extract data (skip the first row with headers)
data = []
for row in rows[1:]:  # Start from the second row onwards
    cols = row.find_all('td')
    cols = [col.text.strip() for col in cols]
    data.append(cols)

# Convert the data into a pandas DataFrame, using the extracted headers as column names
df = pd.DataFrame(data, columns=headers)

# Display the first few rows of the DataFrame to verify
print(df.head())  

# Save the DataFrame to a CSV file
df.to_csv('scraped_data.csv', index=False)