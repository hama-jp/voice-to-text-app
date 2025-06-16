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

import whisper
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
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
    version="1.0.0"
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

class TranscriptionRequest(BaseModel):
    """音声文字起こしリクエスト"""
    use_correction: bool = True
    correction_level: str = "basic"  # "basic" or "advanced"
    language: str = "ja"
    model_size: str = "large-v3"

class TranscriptionResponse(BaseModel):
    """音声文字起こしレスポンス"""
    success: bool
    transcription: str
    corrected_text: Optional[str] = None
    processing_time: float
    corrections_applied: list = []
    file_info: dict = {}

class SaveTextRequest(BaseModel):
    """テキスト保存リクエスト"""
    text: str
    filename: str = "transcription.txt"
    corrected_text: Optional[str] = None

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
        whisper_model = whisper.load_model("large-v3", device=device)
        load_time = time.time() - start_time
        print(f"✅ Whisperモデル読み込み完了 ({load_time:.1f}秒)")
        
        # テキスト校正システム初期化
        print("🔄 テキスト校正システム初期化中...")
        text_corrector = JapaneseTextCorrector(use_llm=True)  # Qwen2.5-7B-Instruct使用
        print("✅ テキスト校正システム初期化完了")
        
        print("🎯 サーバー初期化完了！")
        
    except Exception as e:
        print(f"❌ サーバー初期化エラー: {e}")
        raise e

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "音声テキスト化API",
        "version": "1.0.0",
        "status": "running",
        "whisper_ready": whisper_model is not None,
        "corrector_ready": text_corrector is not None
    }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": {
            "whisper": whisper_model is not None,
            "text_corrector": text_corrector is not None
        }
    }

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    use_correction: bool = True,
    correction_level: str = "basic"
):
    """
    音声ファイルの文字起こし
    
    Args:
        audio_file: 音声ファイル（mp3, wav, m4a等）
        use_correction: テキスト校正を使用するか
        correction_level: 校正レベル（basic/advanced）
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
        
        # 一時ファイル作成
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            tmp.write(await audio_file.read())
            temp_audio_path = tmp.name
        
        # ファイル情報
        file_size = os.path.getsize(temp_audio_path)
        file_info = {
            "filename": audio_file.filename,
            "size_bytes": file_size,
            "size_mb": round(file_size / 1024 / 1024, 2),
            "format": file_extension
        }
        
        print(f"📁 処理開始: {audio_file.filename} ({file_info['size_mb']}MB)")
        print(f"🔧 デバッグ - 受信パラメータ:")
        print(f"   use_correction: {use_correction}")
        print(f"   correction_level: '{correction_level}'")
        print(f"   use_advanced判定: {correction_level == 'advanced'}")
        
        # Whisperで音声認識
        print("🎵 音声認識処理中...")
        whisper_start = time.time()
        
        result = whisper_model.transcribe(
            temp_audio_path,
            language="ja",  # 日本語指定
            task="transcribe",
            fp16=torch.cuda.is_available(),  # GPU利用時はfp16
            temperature=0.0,  # 確定的出力（品質重視）
            beam_size=5,  # ビーム幅拡大（品質向上）
            best_of=5,  # 複数候補から最良選択
            patience=2.0  # より長時間の探索
        )
        
        whisper_time = time.time() - whisper_start
        transcription = result["text"].strip()
        
        print(f"✅ 音声認識完了 ({whisper_time:.2f}秒): {len(transcription)}文字")
        
        # テキスト校正
        corrected_text = None
        corrections_applied = []
        
        if use_correction and text_corrector and transcription:
            print("📝 テキスト校正処理中...")
            print(f"🔧 校正デバッグ:")
            print(f"   文字数: {len(transcription)}文字")
            print(f"   use_advanced: {correction_level == 'advanced'}")
            print(f"   text_corrector.use_llm: {text_corrector.use_llm}")
            correction_start = time.time()
            
            correction_result = text_corrector.correct_text(
                transcription, 
                use_advanced=(correction_level == "advanced")
            )
            
            corrected_text = correction_result["corrected"]
            corrections_applied = correction_result["changes"]
            correction_time = time.time() - correction_start
            
            print(f"✅ テキスト校正完了 ({correction_time:.3f}秒)")
            if corrections_applied:
                print(f"   適用された校正: {', '.join(corrections_applied)}")
        
        # 処理時間計算
        total_time = time.time() - start_time
        
        # レスポンス作成
        response = TranscriptionResponse(
            success=True,
            transcription=transcription,
            corrected_text=corrected_text,
            processing_time=total_time,
            corrections_applied=corrections_applied,
            file_info=file_info
        )
        
        # バックグラウンドでファイル削除
        background_tasks.add_task(cleanup_temp_file, temp_audio_path)
        
        print(f"🎉 処理完了 (総処理時間: {total_time:.2f}秒)")
        
        return response
        
    except Exception as e:
        # エラー時のクリーンアップ
        if temp_audio_path and os.path.exists(temp_audio_path):
            background_tasks.add_task(cleanup_temp_file, temp_audio_path)
        
        print(f"❌ 処理エラー: {e}")
        raise HTTPException(status_code=500, detail=f"処理エラー: {str(e)}")

@app.post("/save_text")
async def save_text_file(request: SaveTextRequest):
    """
    文字起こし結果をテキストファイルとして保存
    
    Args:
        request: SaveTextRequest（text, filename, corrected_text）
    """
    
    try:
        # ファイル名の安全化
        safe_filename = "".join(c for c in request.filename if c.isalnum() or c in "._- ")
        if not safe_filename.endswith('.txt'):
            safe_filename += '.txt'
        
        # 保存内容作成
        content = f"=== 音声文字起こし結果 ===\n"
        content += f"作成日時: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        content += "【元の文字起こし】\n"
        content += request.text + "\n\n"
        
        if request.corrected_text and request.corrected_text != request.text:
            content += "【校正済みテキスト】\n"
            content += request.corrected_text + "\n\n"
        
        # ファイル保存
        output_path = output_dir / safe_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "filename": safe_filename,
            "download_url": f"/download/{safe_filename}",
            "size_bytes": len(content.encode('utf-8'))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ファイル保存エラー: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """ファイルダウンロード"""
    
    file_path = output_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="ファイルが見つかりません")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='text/plain'
    )

@app.get("/supported_formats")
async def get_supported_formats():
    """サポートされている音声形式一覧"""
    return {
        "supported_formats": [".mp3", ".wav", ".m4a", ".flac", ".aac"],
        "max_file_size_mb": 100,  # 仮の制限
        "recommended_format": ".wav",
        "quality_notes": {
            "wav": "無圧縮、最高品質（ファイルサイズ大）",
            "flac": "可逆圧縮、高品質",
            "mp3": "非可逆圧縮、標準品質",
            "m4a": "非可逆圧縮、標準品質",
            "aac": "非可逆圧縮、標準品質"
        }
    }

async def cleanup_temp_file(file_path: str):
    """一時ファイルのクリーンアップ"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            print(f"🗑️  一時ファイル削除: {file_path}")
    except Exception as e:
        print(f"⚠️  一時ファイル削除エラー: {e}")

if __name__ == "__main__":
    print("🎯 音声テキスト化APIサーバーを起動します...")
    print("📋 利用可能なエンドポイント:")
    print("   POST /transcribe - 音声ファイルの文字起こし")
    print("   POST /save_text  - テキストファイル保存")
    print("   GET  /download/{filename} - ファイルダウンロード")
    print("   GET  /health     - ヘルスチェック")
    print("   GET  /supported_formats - サポート形式確認")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )