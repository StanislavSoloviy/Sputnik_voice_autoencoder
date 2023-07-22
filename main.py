from fastapi import FastAPI, File, UploadFile
from Model import Model
import shutil
import os


# Press the green button in the gutter to run the script.
app = FastAPI()

UPLOAD_FOLDER = "uploaded_files"

@app.post("/upload/")
async def upload_files(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Создаем папку, если она не существует
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Сохраняем загруженные файлы
    file1_path = os.path.join(UPLOAD_FOLDER, file1.filename)
    file2_path = os.path.join(UPLOAD_FOLDER, file2.filename)

    with open(file1_path, "wb") as f1, open(file2_path, "wb") as f2:
        shutil.copyfileobj(file1.file, f1)
        shutil.copyfileobj(file2.file, f2)

    model = Model()
    model.create_and_train_model()
    result = model.predict(file1_path, file2_path)
    return {"result": result}






