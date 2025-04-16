import os, csv, requests, logging, glob
from astrapy import DataAPIClient
from pathlib import Path
from dotenv import load_dotenv
from rembg import remove
from PIL import Image

from embedding_extraction import process_image

load_dotenv()

def establish_database_conn():
    """
    Connect to the Astra vector database and get the aba bird images collection or create it if it doesn't exist

    Returns:
        Astra collection
    """

    client = DataAPIClient(os.getenv("ASTRA_DB_APPLICATION_TOKEN"))
    db = client.get_database(os.getenv("ASTRA_DB_API_ENDPOINT"))
    
    pretrained_col_name = "feeder_birds"
    db_cols = db.list_collection_names()

    if (pretrained_col_name not in db_cols):
        return db.create_collection(pretrained_col_name, metric="cosine", dimension=2048)
    else:
        return db.get_collection(pretrained_col_name)
    
def read_csv(sp_name, img_file_limit=100):
    """
    Read csv file for a given species
    
    Args:
        species: species name in correct format (see sp_list.txt)
    Returns:
        List of 100 asset ids from Macaulay Library for that species
    """
    csv_filepath = Path("feeder_birds_csv") / f"{sp_name}.csv"
    
    asset_ids = []
    with open(csv_filepath, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        for idx, row in enumerate(csv_reader, 1):
            asset_ids.append(row[0])

            if idx == img_file_limit:
                break

    return asset_ids
    
def get_images():
    """
    Download images for all species from image filepaths
    """
    with open("feeder_birds_list.txt", 'r') as feeder_birds_list_file:
        for sp_name in feeder_birds_list_file.readlines():
            print(f"Currently downloading {sp_name}")
            sp_name = sp_name.rstrip()
            sp_asset_ids = read_csv(sp_name)
            for i in range(len(sp_asset_ids)):
                img_url = f"https://cdn.download.ams.birds.cornell.edu/api/v2/asset/{sp_asset_ids[i]}/2400"

                try:
                    data = requests.get(img_url, timeout=5).content
                    photo_dir_path = Path(f'feeder_birds/{sp_name}/')
                    os.makedirs(photo_dir_path, exist_ok=True)

                    input_path = photo_dir_path / f"{sp_name}_{str(i).zfill(3)}.jpg"
                    f = open(photo_dir_path / f"{sp_name}_{str(i).zfill(3)}.jpg", 'wb')
                    f = open(input_path, 'wb')
                    f.write(data)
                    f.close()
                
                except requests.exceptions.ReadTimeout as timeout_err:
                    logging.exception(timeout_err)
                    i -= 1

def upload_images(col):
    """
    Upload images to the Astra db collection
    Args:
        - col: Astra db collection
    """    
    img_paths = glob.glob("feeder_birds/**/*.jpg")
    print("Got all image paths")

    batch_size = 100

    print("Processing...")
    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1} with {len(batch)} images")

        for img_path in batch:

            if "augmented" in img_path:
                sp_name = img_path.split('/')[1]
                print(f"uploading {sp_name}")

                img_id = img_path.split('/')[2]

                #img_tensor = process_image(img_path, str(Path("pretrained_model")))
                img_tensor = process_image(img_path)

                print(f"img tensor: {img_tensor}")

                if img_tensor is not None:
                    img_vector = img_tensor.cpu().numpy().flatten()

                    doc_counts = col.count_documents({"_id": img_id}, upper_bound=1)

                    if doc_counts == 0:
                        col.insert_one({
                            "text": sp_name,
                            "_id": img_id,
                            "$vector": img_vector.tolist()
                        })

async def id_image(img_path, col):
    """
    Returns most similar image vectors from database given an image path and database collection
    """
    searching_vector = process_image(img_path).cpu().numpy().flatten()
    most_similar_imgs = col.find(
        sort={"$vector": searching_vector.tolist()},
        limit=5,
        include_similarity=True,
    )

    return most_similar_imgs
