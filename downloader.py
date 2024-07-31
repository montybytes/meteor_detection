import os
import urllib
import requests

from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup


page_url = "https://brams.aeronomie.be/data/downloader?view=downloader"

year = "2024"
month = "05"
day = "26"

fileType = "png"

save_dir = "dataset_temp"


def extract_ids(html_text):
    soup = BeautifulSoup(html_text, "lxml")

    inputTags = [
        inputTag["value"] for inputTag in soup.find(id="systems").find_all("input")
    ]
    inputTags.sort()

    return inputTags


def build_url_dict(stationIds):
    urls = {}

    # select number of stations to download data
    for stationId in stationIds:
        for i in range(0, 24):
            hour = "{}{}".format(0, i) if i < 10 else str(i)
            url = "https://brams.aeronomie.be/downloader.php?"
            params = {
                "type": fileType,
                "system": stationId,
                "year": year,
                "month": month,
                "day": day,
                "hours": hour,
                "minutes": "00",
            }

            key = "-".join([stationId, year, month, day, hour, "00"])
            urls[key] = "{}{}".format(url, urllib.parse.urlencode(params))

    return urls


def download_image(url, save_path):
    response = requests.get(url)
    # only save successful reponses that have an image content
    if response.status_code == 200 and response.headers["Content-Type"] == "image/png":
        with open(save_path, "wb") as f:
            f.write(response.content)

    return


def get_save_path(name):
    filename = "{}.{}".format(name, fileType)

    return os.path.join(save_dir, filename)


def main():
    if os.path.exists(save_dir):
        return

    os.makedirs(save_dir)

    # get html content from url
    page = requests.get(page_url)

    # extract station ids from the html
    stationIds = extract_ids(page.content)

    # build a dict of download urls using the station id as each url's key
    urls = build_url_dict(stationIds)

    # parallelizing download process
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(download_image, value, get_save_path(key))
            for key, value in urls.items()
        ]
        # wait for all downloads to complete
        for future in futures:
            future.result()

    return


if __name__ == "__main__":
    main()
