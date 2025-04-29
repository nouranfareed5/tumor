from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import uvicorn

app = FastAPI(title="Tumor Diagnosis API")

model = tf.keras.models.load_model("model (1).keras")

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224)) 
    image_array = np.array(image) / 255.0 
    image_array = np.expand_dims(image_array, axis=0)  
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        preprocessed = preprocess_image(image_bytes)

        pred = model.predict(preprocessed)
        predicted_index = int(np.argmax(pred))
        class_labels = ['Viable', 'Non-Viable-Tumor', 'Non-Tumor']
        label = class_labels[predicted_index]
        confidence = float(np.max(pred) * 100)

        return JSONResponse({
            "Diagnosis": label,
            "Confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
