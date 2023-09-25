from bs4 import BeautifulSoup
import requests
import json
import re
import pandas as pd
import concurrent
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
import time
import csv


def get_links(pagesToSearch=1):
    """
    Function that gets links of all houses
    """
    s = requests.Session()
    listOfLinks = []
    listOfLinksFinal = []
    #pagesToSearch = 1 # set the number according to how many pages you want to sarch thru each of properties set
    propertiesToSearch = ["apartment","house"] # fill the list with propertis to search

    for prop in propertiesToSearch:
        for x in range(1,pagesToSearch+1):
            search_url = f"https://www.immoweb.be/en/search/{prop}/for-sale?countries=BE&isNewlyBuilt=false&isAPublicSale=false&page={x}&orderBy=relevance"
            r = s.get(search_url)
            soup = BeautifulSoup(r.text,"html.parser")
            for elem in soup.find_all("a", attrs = {"class": "card__title-link"}):
                listOfLinks.append(elem.get('href'))
            listOfLinks = listOfLinks[0:len(listOfLinks)-30]
            listOfLinksFinal = listOfLinksFinal + listOfLinks
            listOfLinks.clear()
    return(listOfLinksFinal)

"""with open('links.txt','w+') as file:
    file.write('\n'.join(links))"""

def get_data(url):
    #url="https://www.immoweb.be/en/classified/house/for-sale/lede/9340/10660142"
    s = requests.Session()
    req= s.get(url)

    soup= BeautifulSoup(req.text, "html.parser")
    data = soup.find('div',attrs={"class":"container-main-content"}).script.text

    Raw_data_InList= re.findall(r"window.classified = (\{.*\})", data)
    Raw_data_InDict = json.loads(Raw_data_InList[0]) # Data dictionary


    property_dict = {"url":url}
    
    try:
        property_dict["locality"] = Raw_data_InDict["customers"][0]['location']['locality']
    except KeyError:
        property_dict["locality"] = 0
    except TypeError:
        property_dict["locality"] = 0
        
    try:
        property_dict["Type_property"] = Raw_data_InDict["property"]["type"]
    except KeyError:
        property_dict["Type_property"] = 0
    except TypeError:
        property_dict["Type_property"] = 0
    try:
        property_dict["subtype_property"] = Raw_data_InDict["property"]["subtype"]
    except KeyError:
        property_dict["subtype_property"] = 0
    except TypeError:
        property_dict["subtype_property"] = 0

    try:
        property_dict["price"] = Raw_data_InDict["price"]["mainValue"]
    except KeyError:
        property_dict["price"] = 0
    except TypeError:
        property_dict["price"] = 0
    
    try:
        property_dict["type-transaction"] = Raw_data_InDict["transaction"]["type"]
    except KeyError:
        property_dict["type-transaction"] = 0
    except TypeError:
        property_dict["type-transaction"] = 0
    
    try:
        property_dict["n_rooms"] = Raw_data_InDict["property"]["bedroomCount"]
    except KeyError:
        property_dict["n_rooms"] = 0
    except TypeError:    
        property_dict["n_rooms"] = 0

    try:
        property_dict["living_area"] = Raw_data_InDict["property"]["netHabitableSurface"]
    except KeyError:
        property_dict["living_area"] = 0
    except TypeError:
        property_dict["living_area"] = 0

    try:
        if bool(Raw_data_InDict["property"]["kitchen"]['type']):
            property_dict["equipped_kitchen"] = 1
        else:
            property_dict["equipped_kitchen"] = 0
    except KeyError:
        property_dict["equipped_kitchen"] = 0
    except TypeError: #added this except for NoneType' object is not subscriptable
        property_dict["equipped_kitchen"] = 0

    try:
        if bool(Raw_data_InDict["transaction"]["sale"]["isFurnished"]):
            property_dict["furnished"] = 1
        else:
            property_dict["furnished"] = 0
    except KeyError:
        property_dict["furnished"] = 0
    except TypeError:
        property_dict["furnished"] = 0

    try:
        if bool(Raw_data_InDict["property"]["fireplaceExists"]):
            property_dict["fireplace"] = 1
        else:
            property_dict["fireplace"] = 0
    except KeyError:
        property_dict["fireplace"] = 0
    except TypeError:
        property_dict["fireplace"] = 0
    
    try:
        if bool(Raw_data_InDict["property"]["hasTerrace"]):
            property_dict["terrace"] = 1
            property_dict["area_terrace"] = Raw_data_InDict["property"]["terraceSurface"]
        else:
            property_dict["terrace"] = 0
            property_dict["area_terrace"] = 0
    except KeyError:
        property_dict["terrace"] = 0
    except TypeError:
        property_dict["terrace"] = 0

    try:
        if bool(Raw_data_InDict["property"]["hasGarden"]):
            property_dict["garden"] = 1
            property_dict["area-garden"] = Raw_data_InDict["property"]["gardenSurface"]
        else:
            property_dict["garden"] = 0
            property_dict["area-garden"] = 0 
    except KeyError:
        property_dict["garden"] = 0
    except TypeError:
        property_dict["garden"] = 0

    try:
        if Raw_data_InDict["property"]["land"]["surface"] != None:
            property_dict["land-surface"] = Raw_data_InDict["property"]["land"]["surface"]
        else:
            property_dict["land-surface"] = 0
    except KeyError:
        property_dict["land-surface"] = 0
    except TypeError: #added this except for NoneType' object is not subscriptable
        property_dict["land-surface"] = 0
    
    # surface area of the plot of land
    
    try:
        property_dict["n-facades"] = Raw_data_InDict["property"]["building"]["facadeCount"]
    except KeyError:
        property_dict["n-facades"] = 0
    except TypeError:
        property_dict["n-facades"] = 0
    
    try:
        if bool(Raw_data_InDict["property"]["hasSwimmingPool"]):
            property_dict["swimming-pool"] = 1
        else:
            property_dict["swimming-pool"] = 0
    except KeyError:
        property_dict["swimming-pool"] = 0
    except TypeError:
        property_dict["swimming-pool"] = 0
        
    try:
        property_dict["state-building"] = Raw_data_InDict["property"]["building"]["condition"]
    except KeyError:
        property_dict["state-building"] = 0
    except TypeError:
        property_dict["state-building"] = 0
            
    return property_dict


data_immo = []
def save(new_data):
        '''This function saves the information acquired from the previous functions and store them in a csv file in the disk.'''
        data_immo.append(new_data)
        dataframe_immo = pd.DataFrame(data_immo)
        dataframe_immo.to_csv("./dataset-immo.csv", index=False, encoding="utf-8")
        print(dataframe_immo)
        return dataframe_immo

start = time.time()
links = get_links(167)
end = time.time()
print("Gathering links time: {:.6f}s".format(end-start))

start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(get_data, link) for link in links]
    for future in concurrent.futures.as_completed(futures):
        save(future.result())        
end = time.time()
print("Time taken for gathering data from {} links: {:.6f}s".format(len(links),end-start))