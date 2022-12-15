import cv2
import numpy as np
import os

#Her defineres stien til datasættet og dens prototext. os gør at den finder stien til hvor den her py fil er placeret.
prototxt_path = os.path.join(os.path.dirname(__file__), "weights/deploy.prototxt.txt") 
model_path = os.path.join(os.path.dirname(__file__), "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel")

model = None

# load Caffe model
def load_model():
    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return model

#faceblur får image fra main, efter det er blevet konverteret fra json i read_imagefile
def faceblur(image):
    global model

    if model is None:
        model = load_model()

    # Henter højde og bredde af image
    h, w = image.shape[:2]
    # gaussian blur koden er afhængig af den kender det originale billedes dimensioner
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1
    # Forbehandler billedet ved at tilpasse dimensionerne på billedet, og laver mean subtraction  
    blob = cv2.dnn.blobFromImage(image, 1.0, (600, 600), (104.0, 177.0, 123.0))

    # Her indsættes det forbehandlet billede ind i det neurale netværk
    model.setInput(blob)
    # output er et numpy array af billedet, hvor alle ansigterne er detekteret
    output = np.squeeze(model.forward())
    
    # I dette array er det en confidence. Nu tjekkes det hvor høj denne confidence er-
    # så kun de ansigter man er nogenlunde sikker på er et ansigt, bliver blurred
    for i in range(0, output.shape[0]):
        confidence = output[i, 2]
        #Her blurres ansigtet
        if confidence > 0.4:
            # Her findes koordinaterne på ansigtet/ansigterne, og opskaleres til- 
            #størrelsen på det originale billede
            box = output[i, 3:7] * np.array([w, h, w, h])
            # Konverter til heltal
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # hent ansigt fra billedet
            face = image[start_y: end_y, start_x: end_x]
            # rediger ansigtet med gausian blur 
            face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
            # indsæt ansigtet på det originale billede  
            image[start_y: end_y, start_x: end_x] = face

    return image

def nparr_to_bytearr(image):
    #Her bliver billedet konverteret til byte array igen
    img_str = cv2.imencode('.jpg', image)[1].tostring()
    type(img_str)
    'str'

    return img_str



def bytearr_to_nparr(file):
    #Her bliver filen konverteret fra byte array til  numpy array
    nparr = np.fromstring(file, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return image

