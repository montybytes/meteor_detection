"""
This python script scrapes the webpage at BRAMS to automate the download process.
This was created to address the inability to mass download the spectrogram images.

Update: this script no longer works as the image endpoints have changed and there
is now a batch download option on the webpage at: https://brams.aeronomie.be/data/downloader
"""

import os
import random
import urllib
import requests
import argparse

from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup


# URL of the webpage to be scraped for downloading spectrogram data
page_url = "https://brams.aeronomie.be/data/downloader?view=downloader"

fileType = "png"

# Setting up the command-line argument parser
parser = argparse.ArgumentParser()

parser.add_argument(
    "date",
    type=str,
    default=None,
    help="The date of the data to be downloaded Format: [DD-MM-YYYY]",
)
parser.add_argument(
    "dir",
    default=None,
    help="The folder to save the data files to",
)

args = parser.parse_args()


def extract_ids(html_text):
    """
    Extracts station IDs from the HTML content of the webpage.

    Args:
        html_text: The raw HTML content of the webpage.

    Returns:
        A sorted list of station IDs.
    """
    # Parse the HTML using lxml parser
    soup = BeautifulSoup(html_text, "lxml")

    # Extracting the values of input tags containing station IDs
    inputTags = [
        inputTag["value"] for inputTag in soup.find(id="systems").find_all("input")
    ]
    inputTags.sort()

    return inputTags


def build_url_dict(stationIds):
    """
    Builds a dictionary of URLs for each station ID and hour of the day.

    Args:
        stationIds: List of station IDs.
    Returns:
        A dictionary where keys are unique identifiers for each station and time, and values are download URLs.
    """
    urls = {}

    # Splitting the date argument into day, month, and year
    date = args.date.split("-")

    # Iterating over each station ID and each hour of the day to build URLs
    for stationId in stationIds:
        for i in range(0, 24):
            hour = (
                "{}{}".format(0, i) if i < 10 else str(i)
            )  # Format hours as "00", "01", ..., "23"
            url = "https://brams.aeronomie.be/downloader.php?"
            params = {
                "type": fileType,
                "system": stationId,
                "year": date[2],
                "month": date[1],
                "day": date[0],
                "hours": hour,
                "minutes": "00",
            }

            # Creating a unique key for each URL
            key = "-".join([stationId, date[2], date[1], date[0], hour, "00"])
            # Encoding the parameters into the URL and storing in the dictionary
            urls[key] = "{}{}".format(url, urllib.parse.urlencode(params))

    return urls


def download_image(url, save_path):
    """
    Downloads an image from the given URL and saves it to the specified path.

    Args:
        url: The URL to download the image from.
        save_path: The file path to save the downloaded image.
    """
    response = requests.get(url)

    # Only save the image if the response is successful and the content type is PNG
    if response.status_code == 200 and response.headers["Content-Type"] == "image/png":
        with open(save_path, "wb") as f:
            # Writing the image content to a file
            f.write(response.content)
    else:
        print("Unable to download from given url: {}".format(url))

    return


def get_save_path(path, name):
    """
    Constructs the full file path for saving the downloaded image.

    Args: 
        path: The directory path where images will be saved.
        name: The name of the file to be saved (without extension).

    Returns: The full file path including the filename and extension.
    """
    filename = "{}.{}".format(
        name, fileType
    )  # Adding the file extension to the filename

    return os.path.join(path, filename)  # Joining the directory path with the filename


def parallel_download(urls, path):
    """
    Downloads images in parallel using multiple threads.

    Args: 
        urls: A dictionary of URLs to download.
        path: The directory path where images will be saved.
    """
    # Parallelizing the download process using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submitting download tasks for each URL
        futures = [
            executor.submit(download_image, url, get_save_path(path, name))
            for name, url in urls.items()
        ]
        # Waiting for all downloads to complete
        for future in futures:
            future.result()
    return


def main():
    # Checking if the specified directory already exists
    if os.path.exists(args.dir):  
        print("This directory already exists")
        return

    # Creating the directory if it doesn't exist
    os.makedirs(args.dir)  

    page = requests.get(page_url)

    # Extract station IDs from the HTML content
    stationIds = extract_ids(page.content)

    # Build a dictionary of download URLs using the station IDs and the provided date
    urls = build_url_dict(stationIds)

    # Download images in parallel and save them to the specified directory
    parallel_download(urls, path=os.path.join(args.dir, "images"))

    return


if __name__ == "__main__":
    main()
