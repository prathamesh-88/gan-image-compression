from fastapi import FastAPI, File, Form, Request, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from hashlib import sha256
from driver import compress, decompress
from PIL import Image
import os
from helper import get_measures

if not os.path.exists("images"):
    os.mkdir("images")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount('/images', StaticFiles(directory='images'), name='images')
exposed_files = {}
image_data = {}

def generate_image(file, image_hash):
    image = Image.open(file)
    image.save("images/" + image_hash + ".png")
    output = decompress(compress(image))
    output_hash = sha256(output.tobytes()).hexdigest()
    output.save("images/" + output_hash + ".png")
    exposed_files[image_hash] = output_hash
    ssim, psnr, mse = get_measures(image_hash, output_hash)
    image_data[image_hash] = {
        "ssim": round(ssim, 2),
        "psnr": round(psnr, 2),
        "mse": round(mse, 5),
    }
    


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
    })


@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, image: UploadFile = File(...)):
    image_hash = sha256(image.file.read()).hexdigest()
    image.file.seek(0)
    background_tasks.add_task(generate_image, image.file, image_hash)
    # redirect to /
    return HTMLResponse(status_code=302, headers={"Location": f"/image/{image_hash}"})


@app.get("/image/{image_hash}")
async def get_image(request: Request, image_hash: str):
    if image_hash in exposed_files:

        return templates.TemplateResponse("image.html", {
            "request": request,
            "input_image": image_hash,
            "output_image": exposed_files[image_hash],
            "data": image_data[image_hash],
        })
    else:
        return templates.TemplateResponse("image.html", {
            "request": request,
            "input_image": image_hash,
            "output_image": None,
            "data": None,
        })