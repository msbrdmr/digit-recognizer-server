from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import base64
import numpy as np
import io
from PIL import Image as imgo
from keras.models import load_model

input_shape = [28, 28, 1]
num_classes = 62

model = load_model("emnist_cnn_model.h5")

app = FastAPI()
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
labels = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(image: dict):
    image_data = image.get("image")
    encoded_image = image_data.split(",")[1]
    img = imgo.open(io.BytesIO(base64.b64decode(encoded_image)))
    img_grayscale = img.convert('L')
    img_to_feed = np.reshape(img_grayscale, (1, 28, 28, 1))
    prediction = model.predict(img_to_feed)
    predicted_label = np.argmax(prediction)
    return {"prediction": int(predicted_label)}


@app.post("/predict_chars")
async def predict(image: dict):
    image_data = image.get("image")
    encoded_image = image_data.split(",")[1]
    img = imgo.open(io.BytesIO(base64.b64decode(encoded_image)))
    img_grayscale = img.convert('L')
    img_resized = img_grayscale.resize((28, 28))
    img_array = np.array(img_resized)
    img_transposed = img_array.transpose()
    # Normalize the image
    img_normalized = img_transposed / 255.0
    img_normalized = img_normalized.astype(np.float32)
    img_normalized = img_normalized.reshape(1, 28, 28, 1)
    print(img_normalized.shape)
    result = model.predict(img_normalized)
    resultLabel = np.argmax(result)

    return {"prediction": labels[resultLabel]}
