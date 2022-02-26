from PIL.Image import Image

from fastapi import FastAPI, HTTPException
from fastapi import UploadFile
from fastapi.responses import StreamingResponse
import os
import numpy as np
from predict import generate_image, generate_latent_space, img_to_array, array_to_img

app = FastAPI()

@app.get("/")
async def index():
    
    return {
        "status": 200,
        "content": {
            "registry": {
                "/compress": {
                    "method": "POST",
                    "description": "Compress an image",
                    "parameters": [
                        {
                            "name": "image",
                            "filetype": "image/png | image/jpeg",
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Compressed image",
                            "filetype": "npy",
                        }
                    }
                },
                "/decompress": {
                    "method": "POST",
                    "description": "Decompress an image",
                    "parameters": [
                        {
                            "name": "compressed image",
                            "filetype": "npy",
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Decompressed image",
                            "filetype": "image/png | image/jpeg",
                        }
                    }
                },
                "/regenerate": {
                    "method": "POST",
                    "description": "Regenerate an image",
                    "parameters": [
                        {
                            "name": "Input Image",
                            "filetype": "image/png | image/jpeg",
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Regenerated image",
                            "filetype": "image/png | image/jpeg",
                        }
                    }
                }

            }
        }
    }


@app.post("/compress")
async def compress(file: UploadFile):
    print("INFO: Received image for compression : {}".format(file.filename))
    img = Image.open(file.file)
    img = img_to_array(img)
    compressed_img = generate_latent_space(img)
    # File name is the same as the original file but with .npy extension and send it back
    return StreamingResponse(compressed_img, media_type="application/octet-stream", headers={"Content-Disposition": "attachment; filename={}".format(file.filename.split(".")[0] + ".npy")})


@app.post("/decompress")
async def decompress(file: UploadFile):
    print("INFO: Received compressed image for decompression : {}".format(file.filename))
    compressed_img = np.load(file.file)
    decompressed_img = generate_image(compressed_img)
    decompressed_img = array_to_img(decompressed_img)
    # File name is the same as the original file but with .png extension and send it back as image with samefile name
    return StreamingResponse(decompressed_img, media_type="image/png", headers={"Content-Disposition": "attachment; filename={}".format(file.filename.split(".")[0] + ".png")})

@app.post("/regenerate")
async def regenerate(file: UploadFile):
    print("INFO: Received compressed image for regeneration : {}".format(file.filename))
    image = Image.open(file.file)
    image = img_to_array(image)
    compressed_img = generate_latent_space(image)

    decompressed_img = generate_image(compressed_img)
    decompressed_img = array_to_img(decompressed_img)
    # File name is the same as the original file but with .png extension and send it back as image with samefile name
    return StreamingResponse(decompressed_img, media_type="image/png", headers={"Content-Disposition": "attachment; filename={}".format(file.filename.split(".")[0] + ".png")})