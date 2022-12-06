from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import StreamingResponse
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

    print("linje 23")
    faceblurring = faceblur(image)
    print("linje 25")
    print(faceblurring)
    return Response(content=faceblurring, media_type="image/jpg")




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
