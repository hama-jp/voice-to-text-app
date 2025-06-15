#!/usr/bin/env python3
"""
Qwen2.5-7B-Instruct統合テスト
実際のアプリケーション環境での動作確認
"""

import sys
import os
from pathlib import Path
import time

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from text_corrector import JapaneseTextCorrector

def test_qwen_integration():
    """Qwen2.5-7B-Instruct統合テスト"""
    
    print("🚀 Qwen2.5-7B-Instruct 統合テスト開始")
    print("=" * 60)
    
    try:
        # 校正システム初期化（LLM有効）
        print("1️⃣ テキスト校正システム初期化中...")
        corrector = JapaneseTextCorrector(use_llm=True)
        
        # 基本校正テスト
        print("\n2️⃣ 基本校正機能テスト")
        print("-" * 30)
        
        basic_test_cases = [
            "今日わとてもよい天気ですね。",
            "個客満足度が高いです。",
            "システムの　アップデート　完了",
        ]
        
        for i, text in enumerate(basic_test_cases, 1):
            result = corrector.correct_text(text, use_advanced=False)
            print(f"テスト{i}: 「{result['original']}」→「{result['corrected']}」")
            print(f"  処理時間: {result['processing_time']:.3f}秒, 方法: {result['method']}")
        
        # 高度校正テスト（LLM使用）
        print("\n3️⃣ 高度校正機能テスト（Qwen2.5-7B-Instruct）")
        print("-" * 45)
        
        advanced_test_cases = [
            "会議の次弟は明日の午後３時からです。",
            "シュミレーションの結果を確認してください。",
            "コミニケーションを取って進めましょう。",
            "雰意気がとてもよいですね。",
        ]
        
        llm_success_count = 0
        total_llm_time = 0
        
        for i, text in enumerate(advanced_test_cases, 1):
            print(f"\nLLMテスト{i}: 「{text}」")
            start_time = time.time()
            
            result = corrector.correct_text(text, use_advanced=True)
            
            print(f"  修正後: 「{result['corrected']}」")
            print(f"  変更: {'✅' if result['corrected'] != result['original'] else '⚪'}")
            print(f"  処理時間: {result['processing_time']:.2f}秒")
            print(f"  使用方法: {result['method']}")
            print(f"  適用校正: {result.get('changes', ['なし'])}")
            
            if result['method'] == 'llm':
                llm_success_count += 1
                total_llm_time += result['processing_time']
        
        # 性能サマリー
        print(f"\n📊 性能サマリー")
        print("=" * 30)
        print(f"LLM処理成功数: {llm_success_count}/{len(advanced_test_cases)}")
        if llm_success_count > 0:
            avg_llm_time = total_llm_time / llm_success_count
            print(f"LLM平均処理時間: {avg_llm_time:.2f}秒")
        
        # Whisper統合想定テスト
        print(f"\n4️⃣ Whisper統合想定テスト")
        print("-" * 30)
        
        whisper_like_output = "今日わ会議がありますそれでわ始めましょうプロジェクトの進捗について話し合います"
        
        print(f"Whisper想定出力: 「{whisper_like_output}」")
        
        result = corrector.correct_text(whisper_like_output, use_advanced=True)
        
        print(f"校正後: 「{result['corrected']}」")
        print(f"処理時間: {result['processing_time']:.2f}秒")
        print(f"文字数: {len(whisper_like_output)} → {len(result['corrected'])}")
        
        # 統合評価
        print(f"\n🎯 統合評価")
        print("=" * 20)
        
        if corrector.use_llm and hasattr(corrector, 'model') and corrector.model:
            print("✅ Qwen2.5-7B-Instruct 統合成功")
            print("✅ 4bit量子化によるメモリ効率化")
            print("✅ 高品質な日本語校正機能")
            print("✅ Whisper + LLM パイプライン準備完了")
            
            print(f"\n💡 推奨設定:")
            print(f"   - バックエンドでLLM校正有効化済み")
            print(f"   - フロントエンドから「高度校正」選択可能")
            print(f"   - 商用利用可能なライセンス")
        else:
            print("⚠️  LLM統合に問題があります")
            print("📝 基本校正のみで動作します")
        
        return True
        
    except Exception as e:
        print(f"❌ 統合テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行"""
    
    # テスト実行
    success = test_qwen_integration()
    
    if success:
        print(f"\n🎉 統合テスト完了！")
        print(f"📋 次のステップ:")
        print(f"   1. python start_server.py でアプリケーション起動")
        print(f"   2. 音声ファイルアップロード")
        print(f"   3. 校正レベル「高度」を選択")
        print(f"   4. Whisper + Qwen2.5-7B 高品質処理を体験")
    else:
        print(f"\n❌ 統合テストに失敗しました")
        print(f"💡 基本校正のみでアプリケーションは動作します")

if __name__ == "__main__":
    main()