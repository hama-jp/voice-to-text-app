#!/usr/bin/env python3
"""
音声テキスト化API サーバー
Whisper + 日本語校正システムによる高品質な音声文字起こし
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from text_corrector import JapaneseTextCorrector

# APIアプリケーション初期化
app = FastAPI(
    title="音声テキスト化API",
    description="Whisperと日本語校正を組み合わせた高品質音声文字起こしサービス",
    version="1.1.0"
)

# CORS設定（フロントエンドとの連携用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切なオリジンを指定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル変数
whisper_model = None
text_corrector = None
upload_dir = project_root / "uploads"
output_dir = project_root / "outputs"

# ディレクトリ作成
upload_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

class TranscriptionResponse(BaseModel):
    """音声文字起こしレスポンス"""
    success: bool
    transcription: str
    corrected_text: Optional[str] = None
    processing_time: float
    corrections_applied: list = []
    file_info: dict = {}

@app.on_event("startup")
async def startup_event():
    """サーバー起動時の初期化"""
    global whisper_model, text_corrector
    
    print("🚀 音声テキスト化APIサーバー起動中...")
    
    try:
        # GPU確認
        if torch.cuda.is_available():
            print(f"✅ GPU利用可能: {torch.cuda.get_device_name(0)}")
            print(f"   GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            device = "cuda"
        else:
            print("⚠️  CPU使用（GPUが利用できません）")
            device = "cpu"
        
        # Whisperモデル読み込み
        print("🔄 Whisperモデル読み込み中...")
        start_time = time.time()
        whisper_model = WhisperModel("large-v3", device=device, compute_type="float16" if device == "cuda" else "int8")
        load_time = time.time() - start_time
        print(f"✅ Whisperモデル読み込み完了 ({load_time:.1f}秒)")
        
        # テキスト校正システムのインスタンス作成
        text_corrector = JapaneseTextCorrector()
        print("✅ テキスト校正システム準備完了")
        
        print("🎯 サーバー初期化完了！")
        
    except Exception as e:
        print(f"❌ サーバー初期化エラー: {e}")
        raise e

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": {
            "whisper": whisper_model is not None,
            "text_corrector_ready": text_corrector is not None
        }
    }

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    use_correction: bool = Query(True),
    correction_model: Optional[str] = Query("rinna/japanese-gpt-neox-small")
):
    """
    音声ファイルの文字起こし
    """
    
    if not whisper_model:
        raise HTTPException(status_code=503, detail="Whisperモデルが利用できません")
    
    start_time = time.time()
    temp_audio_path = None
    
    try:
        # 音声ファイル保存
        file_extension = Path(audio_file.filename).suffix.lower()
        if file_extension not in ['.mp3', '.wav', '.m4a', '.flac', '.aac']:
            raise HTTPException(
                status_code=400, 
                detail=f"サポートされていない音声形式: {file_extension}"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            tmp.write(await audio_file.read())
            temp_audio_path = tmp.name
        
        file_size = os.path.getsize(temp_audio_path)
        file_info = {
            "filename": audio_file.filename,
            "size_bytes": file_size,
            "size_mb": round(file_size / 1024 / 1024, 2),
            "format": file_extension
        }
        
        print(f"📁 処理開始: {audio_file.filename} ({file_info['size_mb']}MB)")
        
        # Whisperで音声認識
        print("🎵 音声認識処理中...")
        whisper_start = time.time()
        
        segments, info = whisper_model.transcribe(
            temp_audio_path,
            language="ja",
            task="transcribe",
            beam_size=5,
        )
        
        transcription = "".join(segment.text for segment in segments)
        
        whisper_time = time.time() - whisper_start
        print(f"✅ 音声認識完了 ({whisper_time:.2f}秒): {len(transcription)}文字")
        
        # テキスト校正
        corrected_text = None
        corrections_applied = []
        
        if use_correction and text_corrector and transcription:
            print(f"📝 テキスト校正処理中 (モデル: {correction_model})...")
            correction_start = time.time()
            
            correction_result = text_corrector.correct_text(
                transcription, 
                model_name=correction_model
            )
            
            corrected_text = correction_result["corrected"]
            corrections_applied = correction_result["changes"]
            correction_time = time.time() - correction_start
            
            print(f"✅ テキスト校正完了 ({correction_time:.3f}秒)")
            if corrections_applied:
                print(f"   適用された校正: {', '.join(corrections_applied)}")
        
        total_time = time.time() - start_time
        
        response = TranscriptionResponse(
            success=True,
            transcription=transcription,
            corrected_text=corrected_text,
            processing_time=total_time,
            corrections_applied=corrections_applied,
            file_info=file_info
        )
        
        background_tasks.add_task(cleanup_temp_file, temp_audio_path)
        
        print(f"🎉 処理完了 (総処理時間: {total_time:.2f}秒)")
        
        return response
        
    except Exception as e:
        if temp_audio_path and os.path.exists(temp_audio_path):
            background_tasks.add_task(cleanup_temp_file, temp_audio_path)
        
        print(f"❌ 処理エラー: {e}")
        raise HTTPException(status_code=500, detail=f"処理エラー: {str(e)}")

async def cleanup_temp_file(file_path: str):
    """一時ファイルのクリーンアップ"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            print(f"🗑️  一時ファイル削除: {file_path}")
    except Exception as e:
        print(f"⚠️  一時ファイル削除エラー: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )