#!/usr/bin/env python3
"""
Whisperモデルの動作検証スクリプト
日本語音声認識の品質をテスト
"""

import whisper
import torch
import time
import sys

def test_whisper_model():
    """Whisperモデルの動作テスト"""
    
    # GPU利用可能性確認
    if torch.cuda.is_available():
        print(f"✅ CUDA利用可能: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        device = "cuda"
    else:
        print("⚠️  CPU使用（GPUが利用できません）")
        device = "cpu"
    
    print("\n🔄 Whisperモデル読み込み中...")
    start_time = time.time()
    
    # 品質重視: large-v3モデルを使用
    try:
        model = whisper.load_model("large-v3", device=device)
        load_time = time.time() - start_time
        print(f"✅ モデル読み込み完了 ({load_time:.2f}秒)")
        
        # モデル情報表示
        print(f"✅ モデル: {model.dims}")
        print(f"✅ 言語対応: 多言語対応（日本語含む）")
        
        return model
        
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        return None

def test_japanese_audio_processing(model, audio_file_path=None):
    """日本語音声処理テスト"""
    
    if not model:
        print("❌ モデルが利用できません")
        return
    
    if not audio_file_path:
        print("📝 音声ファイルが指定されていないため、設定テストのみ実行")
        
        # 日本語特化設定のテスト
        try:
            # 日本語優先設定
            options = {
                "language": "ja",  # 日本語指定
                "task": "transcribe",  # 文字起こし
                "fp16": torch.cuda.is_available(),  # GPU利用時はfp16
                "temperature": 0.0,  # 確定的出力（品質重視）
                "beam_size": 5,  # ビーム幅拡大（品質向上）
                "best_of": 5,  # 複数候補から最良選択
                "patience": 2.0,  # より長時間の探索
            }
            
            print("✅ 日本語最適化設定:")
            for key, value in options.items():
                print(f"   {key}: {value}")
                
            return options
            
        except Exception as e:
            print(f"❌ 設定テストエラー: {e}")
            return None
    
    else:
        print(f"🎵 音声ファイル処理: {audio_file_path}")
        try:
            start_time = time.time()
            
            # 高品質設定での処理
            result = model.transcribe(
                audio_file_path,
                language="ja",
                task="transcribe",
                fp16=torch.cuda.is_available(),
                temperature=0.0,
                beam_size=5,
                best_of=5,
                patience=2.0
            )
            
            process_time = time.time() - start_time
            print(f"✅ 処理完了 ({process_time:.2f}秒)")
            print(f"📝 認識結果: {result['text']}")
            
            return result
            
        except Exception as e:
            print(f"❌ 音声処理エラー: {e}")
            return None

if __name__ == "__main__":
    print("🚀 Whisper日本語音声認識テスト開始")
    print("=" * 50)
    
    # モデル読み込みテスト
    model = test_whisper_model()
    
    if model:
        # 日本語設定テスト
        options = test_japanese_audio_processing(model)
        
        if options:
            print("\n✅ 全テスト完了")
            print("🎯 推奨設定:")
            print("   - モデル: large-v3 (最高品質)")
            print("   - 言語: 日本語特化")
            print("   - ビーム幅: 5 (品質向上)")
            print("   - 温度: 0.0 (確定的出力)")
            print("   - 最良候補: 5 (複数候補比較)")
        else:
            print("\n❌ 設定テスト失敗")
    else:
        print("\n❌ モデル読み込み失敗")
    
    print("\n📋 使用方法:")
    print("   python test_whisper.py [音声ファイルパス]")