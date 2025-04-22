import requests
import zipfile

INDEX_URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"

def download_zip(url, output_path):
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            print("file downloaded")
    except requests.exceptions.RequestException as e:
        print(f"error during {e}")

def extract_zip(zip_path, save_dir):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip:
            zip.extractall(save_dir)
        print("file saved successfully")
    except zipfile.BadZipFile as e:
        print("error duing unzip")


download_zip(INDEX_URL, "data/comments.zip")
extract_zip("comments.zip", "data/raw")
