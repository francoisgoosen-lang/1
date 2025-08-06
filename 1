from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import librosa
import tempfile

app = FastAPI()

@app.post("/analyze-bpm")
async def analyze_bpm(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return JSONResponse(content={"bpm": round(tempo, 2)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
