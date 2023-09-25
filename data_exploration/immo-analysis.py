import requests
from bs4 import BeautifulSoup
import pandas as pd

# Scraping the information of province and post code from this link
url = 'https://www.metatopos.eu/belgcombiN.html'

post_code = []
city = []
subcities = []
province = []
r = requests.get(url)
soup = BeautifulSoup(r.content, 'html.parser')

# appending the first 4 informations from the 7 columns presented in a row in the correspondent list
content = soup.find_all("td")
for i in range(0, len(content), 7):
    post_code.append(content[i].text)
    city.append(content[i+1].text)
    subcities.append(content[i+2].text)
    province.append(content[i+3].text)

# removing the headers from the website
post_code = post_code[1:]
city = city[1:]
subcities = subcities[1:]
province = province[1:]

# creating a dataframe and saving it in a csv file
df = pd.DataFrame({"post_code": post_code, "city": city, "province": province})
df.to_csv("provinces.csv", index=False)

# assigning the 2 datasets into variables in order to be merged
df = pd.read_csv("data/dataset-immo.csv")
df2 = pd.read_csv("data/provinces.csv")

df['locality'] = df['locality'].astype(str)
df2['city'] = df2['city'].astype(str)

# merging the information of province and post code from the scrapped website to the dataset-immo.csv
merged_data = pd.merge(df, df2[["city", "province"]], left_on="locality", right_on="city", how="left")
df["province"] = merged_data["province"]

# saving the changes on the dataset-immo.csv'''
merged_data.to_csv("with_provinces_dataset.csv", index=False)