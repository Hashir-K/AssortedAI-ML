import requests
from bs4 import BeautifulSoup
import os

url = 'https://example.com/documents'  # Replace with the actual URL of the webpage
response = requests.get(url)

# Verify the request was successful
if response.status_code == 200:
    print('Successfully retrieved the webpage.')
else:
    print('Failed to retrieve the webpage. Status code:', response.status_code)

soup = BeautifulSoup(response.content, 'html.parser')

# Optional: Print the title of the webpage to confirm successful parsing
print('Webpage Title:', soup.title.text)

# Locate the <a> tag that contains the link to the document
document_link = soup.find('a', {'class': 'download-link'})['href']  # Replace with the actual class or identifier

# Print the document link to verify
print('Document link found:', document_link)

base_url = 'https://example.com'  # The base URL of the website
full_url = os.path.join(base_url, document_link)

print('Full URL:', full_url)

# Send a GET request to download the document
document_response = requests.get(full_url)

# Check if the document request was successful
if document_response.status_code == 200:
    # Save the document to a file
    with open('document.pdf', 'wb') as file:  # Replace 'document.pdf' with the appropriate filename and extension
        file.write(document_response.content)
    print('Document downloaded successfully.')
else:
    print('Failed to download the document. Status code:', document_response.status_code)


#this is for handling multiple documents; i.e. If the webpage contains multiple documents, 
# you can modify the scraper to loop through all available document links and download each one:

# Find all <a> tags with the document links
document_links = soup.find_all('a', {'class': 'download-link'})  # Replace with the actual class or identifier

# Loop through each link and download the corresponding document
for i, link in enumerate(document_links):
    document_url = os.path.join(base_url, link['href'])
    document_response = requests.get(document_url)
    
    if document_response.status_code == 200:
        # Save each document with a unique name
        file_name = f'document_{i+1}.pdf'  # Adjust the file name as needed
        with open(file_name, 'wb') as file:
            file.write(document_response.content)
        print(f'Document {i+1} downloaded successfully as {file_name}.')
    else:
        print(f'Failed to download document {i+1}. Status code:', document_response.status_code)