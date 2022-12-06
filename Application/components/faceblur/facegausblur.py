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

    # get width and height of the image
    h, w = image.shape[:2]
    # gaussian blur kernel size depends on width and height of original image
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1
    # preprocess the image: resize and performs mean subtraction
    blob = cv2.dnn.blobFromImage(image, 1.0, (600, 600), (104.0, 177.0, 123.0))
    # set the image into the input of the neural network
    model.setInput(blob)
    # perform inference and get the result
    output = np.squeeze(model.forward())
    for i in range(0, output.shape[0]):
        confidence = output[i, 2]
        # get the confidence
        # if confidence is above 40%, then blur the bounding box (face)
        if confidence > 0.4:
            # get the surrounding box cordinates and upscale them to original image
            box = output[i, 3:7] * np.array([w, h, w, h])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # get the face image
            face = image[start_y: end_y, start_x: end_x]
            # apply gaussian blur to this face
            face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
            # put the blurred face into the original image
            image[start_y: end_y, start_x: end_x] = face
    

    #Her bliver billedet konverteret til json
    img_str = cv2.imencode('.jpg', image)[1].tostring()
    type(img_str)
    'str'
    #print("end of faceblur")
    

    return img_str

def read_imagefile(file):
    #Her bliver filen konverteret fra json til array form, som cv2 kan forstå
    print(file)
    nparr = np.fromstring(file, np.uint8)
    
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print("end of read_imagefile")


    return image
