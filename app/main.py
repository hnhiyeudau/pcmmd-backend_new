from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.services.predict import predict_image

app = FastAPI()


# Cho phép gọi từ mọi nguồn (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pcmmd-frontend.vercel.app"],  # hoặc chỉ định domain frontend của bạn
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# print("✅ CORS middleware loaded with:", allow_origins)

# Root endpoint kiểm tra server
@app.get("/")
def root():
    return {"message": "PCMMD API is live 🎉"}

# API dự đoán ảnh
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    result = await predict_image(file)
    return result
