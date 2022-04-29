from argparse import ArgumentParser
from pathlib import Path
import requests, zipfile, io
from tqdm.auto import tqdm
import pandas as pd


URL_FILE = Path(__file__).parent / "dataset_urls.txt"

number_img_balls = 0
number_img_no_balls = 0

def get_url_list(filename: str) -> list:
    with open(filename) as fp:
        return [url for url in fp.read().split('\n') if url]


def download_zip(url: str) -> zipfile.ZipFile:
    CHUNK_SIZE = 16

    with requests.get(url=url, stream=True) as response:
        if response.ok:
            total_length = int(response.headers.get("Content-Length"))

            content = io.BytesIO()

            with tqdm.wrapattr(response.raw, "read", total=total_length, desc="") as raw:
                while True:
                    chunk = raw.read(CHUNK_SIZE)
                    content.write(chunk)

                    if not chunk:
                        break

            return zipfile.ZipFile(content)
        else:
            print("Could not download a zip file...")
            print(f"URL ({url}) responded with status code: {response.status_code}")
            return None


def _extract_img_balls(zip_file: zipfile.ZipFile, annotation):
    global number_img_balls

    filenames = annotation.filename

    for idx, filename in enumerate(filenames):
        file_extension = filename[filename.rfind('.') + 1:]
        img_path = f"train/{filename}"
        img = zip_file.read(img_path)

        new_filename = f"{number_img_balls}.{file_extension}"
        annotation.filename[idx] = new_filename
        number_img_balls += 1

        with open(BALLS_DIR / new_filename, "wb") as ifp:
            ifp.write(img)

    header = False if ANNOTATION_FILENAME.exists() else True
    annotation.to_csv(
        ANNOTATION_FILENAME, 
        mode='a', 
        index=False, 
        header=header
    )


def _extract_img_no_balls(zip_file: zipfile.ZipFile, filenames):
    global number_img_no_balls

    for filename in filenames:
        file_extension = filename[filename.rfind('.') + 1:]

        if file_extension not in ["png", "jpg", "jpeg"]:
            continue

        img = zip_file.read(filename)
        
        new_filename = NO_BALLS_DIR / f"{number_img_no_balls}.{file_extension}"
        number_img_no_balls += 1

        with open(new_filename, "wb") as ifp:
            ifp.write(img)


def extract_imgs(zip_file: zipfile.ZipFile):
    annotation_filename = f"train/{ANNOTATION_FILENAME.name}"

    if annotation_filename not in zip_file.namelist():
        print("Cannot find annotation file - skipping...")
        return

    nones_filenames = set(zip_file.namelist())

    with zip_file.open(annotation_filename) as afp:
        annotation = pd.read_csv(afp, index_col=False)

        nones_filenames -= set(annotation.filename)

        _extract_img_balls(zip_file, annotation)
        _extract_img_no_balls(zip_file, nones_filenames)


parser = ArgumentParser()
parser.add_argument(
    "-u", "--url-file", 
    type=str, 
    help="Path for a URL text file",
    default=URL_FILE
)
parser.add_argument(
    "-d", "--dir",
    type=str,
    help="A directory where dataset will be placed",
    default=Path(__file__).parent / "dataset"
)


if __name__ == "__main__":

    args = parser.parse_args()

    DATASET_DIR = Path(args.dir)
    BALLS_DIR = DATASET_DIR / "balls"
    NO_BALLS_DIR = DATASET_DIR / "no_balls"
    ANNOTATION_FILENAME = BALLS_DIR / "_annotations.csv"

    if not BALLS_DIR.exists():
        BALLS_DIR.mkdir(parents=True)

    if not NO_BALLS_DIR.exists():
        NO_BALLS_DIR.mkdir(parents=True)

    url_list = get_url_list(args.url_file)

    if not url_list:
        print("Nothing to download from")
    else:
        for url in url_list:
            try:
                zip_file = download_zip(url)
                extract_imgs(zip_file)
            except Exception() as e:
                print(f"Cannot download from {url}")
                print(e)
                
    print(f"{number_img_balls} images with balls")
    print(f"{number_img_no_balls} images without balls")
