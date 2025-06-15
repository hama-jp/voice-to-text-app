#!/usr/bin/env python3
"""
音声テキスト化Webアプリケーション 起動スクリプト
"""

import os
import sys
import time
import threading
import webbrowser
from pathlib import Path

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def start_backend():
    """バックエンドサーバー起動"""
    print("🚀 バックエンドサーバーを起動中...")
    
    # 仮想環境の確認
    venv_path = project_root / "venv"
    if not venv_path.exists():
        print("❌ 仮想環境が見つかりません")
        print("💡 以下のコマンドで仮想環境を作成してください:")
        print("   python3 -m venv venv")
        print("   source venv/bin/activate  # Linux/Mac")
        print("   venv\\Scripts\\activate.bat  # Windows")
        print("   pip install -r requirements.txt")
        return False
    
    # バックエンド起動
    try:
        import uvicorn
        from backend.main import app
        
        print("✅ バックエンドサーバー準備完了")
        print("📡 サーバー起動: http://localhost:8000")
        
        # サーバー起動（別スレッドで）
        def run_server():
            uvicorn.run(
                app,
                host="127.0.0.1",
                port=8000,
                log_level="warning"  # ログレベル下げる
            )
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        return True
        
    except ImportError as e:
        print(f"❌ 必要なパッケージが不足しています: {e}")
        print("💡 以下のコマンドで依存関係をインストールしてください:")
        print("   pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ サーバー起動エラー: {e}")
        return False

def check_server_ready(max_attempts=30):
    """サーバー起動完了を確認"""
    import requests
    
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://127.0.0.1:8000/health", timeout=1)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
        print(f"⏳ サーバー起動待機中... ({attempt + 1}/{max_attempts})")
    
    return False

def start_frontend():
    """フロントエンド（ブラウザ）起動"""
    print("🌐 Webブラウザでアプリケーションを開きます...")
    
    frontend_file = project_root / "frontend" / "index.html"
    if not frontend_file.exists():
        print("❌ フロントエンドファイルが見つかりません")
        return False
    
    # ブラウザでHTMLファイルを開く
    try:
        webbrowser.open(f"file://{frontend_file.absolute()}")
        print("✅ ブラウザでアプリケーションを開きました")
        return True
    except Exception as e:
        print(f"❌ ブラウザ起動エラー: {e}")
        print(f"💡 手動で以下のファイルを開いてください:")
        print(f"   {frontend_file.absolute()}")
        return False

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("🎵 音声テキスト化Webアプリケーション")
    print("=" * 60)
    print()
    
    # 必要ファイルの確認
    required_files = [
        project_root / "backend" / "main.py",
        project_root / "frontend" / "index.html",
        project_root / "text_corrector.py"
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        print("❌ 必要なファイルが不足しています:")
        for f in missing_files:
            print(f"   {f}")
        return
    
    # バックエンド起動
    if not start_backend():
        print("❌ バックエンドサーバーの起動に失敗しました")
        return
    
    # サーバー起動完了待機
    print("⏳ サーバー初期化中...")
    if not check_server_ready():
        print("❌ サーバー起動タイムアウト")
        return
    
    print("✅ サーバー起動完了!")
    
    # フロントエンド起動
    start_frontend()
    
    print()
    print("🎯 アプリケーション起動完了!")
    print("📋 利用方法:")
    print("   1. ブラウザで音声ファイル（MP3, WAV等）をアップロード")
    print("   2. 「音声を文字起こしする」ボタンをクリック")
    print("   3. 文字起こし結果を確認・ダウンロード")
    print()
    print("⚙️  設定:")
    print("   - テキスト校正: ONで誤字訂正・文章改善")
    print("   - 校正レベル: 基本（高速）/高度（精密）")
    print()
    print("🌐 アクセス先:")
    print(f"   フロントエンド: file://{project_root / 'frontend' / 'index.html'}")
    print("   API: http://127.0.0.1:8000")
    print("   APIドキュメント: http://127.0.0.1:8000/docs")
    print()
    print("❌ 終了するには Ctrl+C を押してください")
    
    try:
        # サーバーを動かし続ける
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 アプリケーションを終了します...")

if __name__ == "__main__":
    main()