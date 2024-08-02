import os
import random
import urllib
import requests
import argparse

from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup


page_url = "https://brams.aeronomie.be/data/downloader?view=downloader"

fileType = "png"

# setting up command line argument parser
parser = argparse.ArgumentParser()

parser.add_argument(
    "-s",
    "--sort",
    default=True,
    action="store_true",
    help="Sort files into train and validation folders [Default: true]",
)
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

sort_mode = args.sort


def sort_files():
    return


def extract_ids(html_text):
    soup = BeautifulSoup(html_text, "lxml")

    inputTags = [
        inputTag["value"] for inputTag in soup.find(id="systems").find_all("input")
    ]
    inputTags.sort()

    return inputTags


def build_url_dict(stationIds):
    urls = {}

    date = args.date.split("-")

    # select number of stations to download data
    for stationId in stationIds:
        for i in range(0, 24):
            hour = "{}{}".format(0, i) if i < 10 else str(i)
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

            key = "-".join([stationId, date[2], date[1], date[0], hour, "00"])
            urls[key] = "{}{}".format(url, urllib.parse.urlencode(params))

    return urls


def download_image(url, save_path):
    response = requests.get(url)
    # only save successful reponses that have an image content
    if response.status_code == 200 and response.headers["Content-Type"] == "image/png":
        with open(save_path, "wb") as f:
            f.write(response.content)
    else:
        print("Unable to download from given url: {}".format(url))

    return


def get_save_path(path, name):
    filename = "{}.{}".format(name, fileType)

    return os.path.join(path, filename)


def parallel_download(urls, path):
    # parallelizing download process
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(download_image, url, get_save_path(path, name))
            for name, url in urls.items()
        ]
        # wait for all downloads to complete
        for future in futures:
            future.result()
    return


def main():
    if os.path.exists(args.dir):
        print("This directory already exists")
        return

    os.makedirs(args.dir)

    # get html content from url
    page = requests.get(page_url)

    # extract station ids from the htmlww
    stationIds = extract_ids(page.content)

    # build a dict of download urls using the station id as each url's key
    urls = build_url_dict(stationIds)

    if sort_mode:
        urls = list(urls.items())
        random.shuffle(urls)
        split_index = int(len(urls) * 0.8)

        train_dir = os.path.join(args.dir, "images", "train")
        val_dir = os.path.join(args.dir, "images", "val")

        os.makedirs(train_dir)
        os.makedirs(val_dir)

        train_urls, val_urls = dict(urls[:split_index]), dict(urls[split_index:])

        parallel_download(train_urls, train_dir)
        parallel_download(val_urls, val_dir)

    else:
        parallel_download(urls, path=os.path.join(args.dir, "images"))

    return


if __name__ == "__main__":
    main()
