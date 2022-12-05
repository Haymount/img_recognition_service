from fastapi import FastAPI, UploadFile, File
import uvicorn
from starlette.responses import RedirectResponse


from components import faceblur, read_imagefile


app = FastAPI()

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post('/image/gausblur')
async def gausblur(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
   
    image = read_imagefile(await file.read())
    faceblurring = faceblur(image)

    return faceblurring




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
