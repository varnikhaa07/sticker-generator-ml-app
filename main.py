from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from rembg import remove
from PIL import Image
import numpy as np
import io
import cv2

app = FastAPI()

@app.post("/remove-bg/")
async def remove_bg(file: UploadFile = File(...)):
    contents = await file.read()

    input_image = Image.open(io.BytesIO(contents)).convert("RGBA")
    output_image = remove(input_image)

    img = np.array(output_image)
    alpha = img[:, :, 3]

    _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)

    kernel = np.ones((10, 10), np.uint8)
    border = cv2.dilate(mask, kernel, iterations=1)
    border_only = cv2.subtract(border, mask)

    border_img = np.zeros_like(img)
    border_img[:, :, :3] = 255
    border_img[:, :, 3] = border_only

    combined = border_img.copy()
    alpha_mask = img[:, :, 3] > 0
    combined[alpha_mask] = img[alpha_mask]

    final_image = Image.fromarray(combined)

    img_byte_arr = io.BytesIO()
    final_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return Response(content=img_byte_arr.getvalue(), media_type="image/png")