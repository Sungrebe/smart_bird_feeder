from create_database import establish_database_conn, upload_images, id_image
from embedding_extraction import process_image
import asyncio

from matplotlib import pyplot as plt
from matplotlib import image as mp_img

def display_img(image_path, index, rows=1, columns=2, text=""):
    """
    Display an image
    """
    plt.subplot(rows, columns, index)
    img = mp_img.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{image_path}")
    plt.text(0.5, -0.1, text, ha='center', transform=plt.gca().transAxes)
    plt.savefig('myplot.png')

async def main():
    #print("Setting up collection")
    col = establish_database_conn()
    #print("Setup complete")
    #get_images()
    #upload_images(col)
    #create_triplets()
    plt.figure(figsize=(4, 5))

    img_path = "image.png"
    display_img(img_path, 1)
    # # print("Getting results...")
    results = await id_image(img_path, col)
    # # print("Received results")

    for result in results:
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
