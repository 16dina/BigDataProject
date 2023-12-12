from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image as PILImage
from fastcore.foundation import Config
from fastai.learner import load_learner
from io import BytesIO
from PIL import Image
import io


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = FastAPI()
#config = Config()
learner = load_learner('export.pkl')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

import numpy as np



@app.post('/predict')
async def predict(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail='No image found in request')

    file = files[0]
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail='File extension not allowed')
    


    try:
        content = await file.read()
        img = PILImage.open(BytesIO(content))
        
        # Check if the image opened successfully
        if img is None:
            raise HTTPException(status_code=400, detail='Failed to open image')
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGB')

        # Convert the processed image back to bytes in memory
        buffered = BytesIO()
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        if file_extension == 'png':
            img.save(buffered, format="PNG")
        else:
            img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        pred = learner.predict(img)
        return JSONResponse(content=str(pred))
        # Return the processed image as a response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error processing image: {str(e)}')

if __name__ == '__main__':
    import uvicorn
    #port = config.get('PORT', 5000)
    uvicorn.run(app, host='0.0.0.0', port=5000)