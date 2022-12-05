from fastapi import FastAPI, UploadFile, File
import uvicorn
from starlette.responses import RedirectResponse

from application.components import faceblur, read_imagefile


app = FastAPI()

@app.get("/index")
def hello_world(name:str):
    return f"Hello {name}!"


@app.post('/anon/gausblur')
async def gausblur(file: UploadFile = File(...)):
    extension = file.filename.split(.)[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
   
    image = read_imagefile(await file.read)
    
    faceblur = preprocess(image)

    return faceblur




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
