from PIL.Image import Image
from fastapi import BackgroundTasks, background
from fastapi import FastAPI, HTTPException
import os
from random import randbytes

from fastapi import UploadFile, File
from fastapi.responses import FileResponse

app = FastAPI()


async def run_model(model: str, img: Image, token: str = ''):
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    import numpy as np
    if model + '.h5' in os.listdir(os.path.join(os.getcwd(), 'models')):
        runner = load_model(os.path.join(os.getcwd(), 'models', model + '.h5'))
        result = runner.predict(img_to_array(img))
        np.save(os.path.join(os.getcwd(), 'results', token + '.npy'), result)
        




@app.get("/models/")
async def models():
    models = [model.split('.')[0] for model in os.listdir(os.path.join(os.getcwd, 'models'))]
    return {
        'models': models
    }

@app.post("/predict/{model}")
async def predict(model: str, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    from io import StringIO
    from PIL import Image
    img = Image.open(file.file)
    token = randbytes(16).hex()
    background_tasks.add_task(run_model, model + '.h5', img, token=token)
    return { 'token': token }


@app.get("/prediction{token}", response_class=FileResponse)
async def prediction(token: str):
    if token + '.npy' in os.listdir(os.path.join(os.getcwd(), 'results')):
        return os.path.join(os.getcwd(), 'results', token + '.npy')
    else:
        raise HTTPException(status_code=404, detail="The result isn't computed")
    