from fastapi import FastAPI, UploadFile, File, Response
from starlette.responses import RedirectResponse    
from components import faceblur, bytearr_to_nparr, nparr_to_bytearr
import uvicorn

app = FastAPI()

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

#Dette er vores faceblur endpoint
@app.post('/image/gausblur')
async def gausblur(file: UploadFile = File(...)):
    #Her tjekkes det om den content der bliver modtaget er den rigtige type
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg")
    if not extension:
        return "Image must be jpg or png format!"
    
    #Her læses filen, og konverteres fra et byte array-
    #til et numpy array, som faceblur() kan modtage
    image = bytearr_to_nparr(await file.read())

    #Her kalder vi den funktion som skal blur billedet.
    faceblurred = faceblur(image)

    #Her konverteres billedet tilbage til et byte array
    faceblurred_str = nparr_to_bytearr(faceblurred)


    #Her returnerer vi det anonymiseret billede til klienten- 
    #-i form af en json string. faceblur() returnerer kun i jpg format.
    return Response(content=faceblurred_str, media_type="image/jpg")




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
