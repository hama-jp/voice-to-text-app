#!/usr/bin/env python3
"""
軽量テキスト校正システムのテスト（LLMなし）
基本的な誤字訂正と正規化機能の検証
"""

import sys
import os

# パスの追加
sys.path.append(os.path.dirname(__file__))

from text_corrector import JapaneseTextCorrector

def test_basic_correction():
    """基本校正機能のテスト"""
    
    print("📝 基本テキスト校正機能テスト")
    print("=" * 50)
    
    # LLMなしで初期化（基本機能のみ）
    corrector = JapaneseTextCorrector(use_llm=False)
    
    test_cases = [
        "今日わとてもよい天気ですね。",
        "会議の次弟は明日の午後３時からです。",
        "この商品の個客満足度は高いです。",
        "シュミレーションの結果を確認してください。",
        "お疲れ様でした！！！　　　",
        "こんにちわ、いかがお過ごしですか？？？",
        "コミニケーションが大切です。",
        "バッテリの交換が必要です。",
        "雰意気がよくないですね。",
    ]
    
    total_improvements = 0
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n--- テスト {i} ---")
        result = corrector.correct_text(test_text, use_advanced=False)
        
        print(f"修正前: 「{result['original']}」")
        print(f"修正後: 「{result['corrected']}」")
        
        if result['corrected'] != result['original']:
            print(f"✅ 改善: {', '.join(result['changes'])}")
            total_improvements += 1
        else:
            print("⚪ 変更なし")
        
        print(f"処理時間: {result['processing_time']:.3f}秒")
    
    print(f"\n📊 テスト結果サマリー")
    print(f"総テスト数: {len(test_cases)}")
    print(f"改善された文: {total_improvements}")
    print(f"改善率: {total_improvements/len(test_cases)*100:.1f}%")
    
    return corrector

def test_whisper_integration():
    """Whisper出力との統合テスト想定"""
    
    print(f"\n🎵 Whisper出力想定テスト")
    print("=" * 30)
    
    corrector = JapaneseTextCorrector(use_llm=False)
    
    # Whisperで起こりがちな認識エラーパターン
    whisper_like_errors = [
        "今日は会議がありますそれでは始めましょう",  # 句読点なし
        "プロジェクトの　　　進捗　　を　確認します",  # 不規則な空白
        "来週の金曜日に発表会があります　ご参加ください",  # 文境界不明確
        "システムの　アップデート　が　完了しました",  # 不自然な空白
    ]
    
    for i, text in enumerate(whisper_like_errors, 1):
        print(f"\nWhisper出力想定 {i}:")
        result = corrector.correct_text(text)
        
        print(f"修正前: 「{result['original']}」")
        print(f"修正後: 「{result['corrected']}」")
        
        if result['changes']:
            print(f"適用処理: {', '.join(result['changes'])}")

def benchmark_performance():
    """パフォーマンステスト"""
    
    print(f"\n⚡ パフォーマンステスト")
    print("=" * 30)
    
    corrector = JapaneseTextCorrector(use_llm=False)
    
    # 様々な長さのテキスト
    test_texts = [
        "短いテスト",  # 短文
        "これは中程度の長さの文章です。今日わとてもよい天気ですね。会議の次弟は明日です。",  # 中文
        "これはより長い文章のテストです。" * 10,  # 長文
    ]
    
    for i, text in enumerate(test_texts, 1):
        result = corrector.correct_text(text)
        
        print(f"テスト {i} (文字数: {len(text)})")
        print(f"  処理時間: {result['processing_time']:.3f}秒")
        print(f"  文字/秒: {len(text)/result['processing_time']:.0f}")

if __name__ == "__main__":
    try:
        # 基本機能テスト
        corrector = test_basic_correction()
        
        # Whisper統合テスト
        test_whisper_integration()
        
        # パフォーマンステスト
        benchmark_performance()
        
        print(f"\n✅ 全テスト完了")
        print(f"🎯 基本校正システムが正常に動作しています")
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()