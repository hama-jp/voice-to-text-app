#!/usr/bin/env python3
"""
Qwen2.5-7B-Instructモデルを使用した日本語校正システムのテスト
4bit量子化による高効率実装
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import gc

class QwenTextCorrector:
    """Qwen2.5-7B-Instructを使用した日本語テキスト校正システム"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        """
        初期化
        
        Args:
            model_name (str): 使用するQwenモデル名
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🤖 Qwen日本語校正システム初期化: {model_name}")
        self._initialize_model()
    
    def _initialize_model(self):
        """4bit量子化でQwenモデルを初期化"""
        try:
            print("🔄 Qwenモデル読み込み中...")
            start_time = time.time()
            
            # 4bit量子化設定（最適化）
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # トークナイザー読み込み
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # モデル読み込み（4bit量子化）
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            load_time = time.time() - start_time
            print(f"✅ Qwenモデル読み込み完了 ({load_time:.1f}秒)")
            
            # メモリ使用量確認
            if self.device == "cuda":
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"   GPU メモリ使用量: {memory_used:.1f}GB")
            
            return True
            
        except Exception as e:
            print(f"❌ Qwenモデル読み込みエラー: {e}")
            self.model = None
            self.tokenizer = None
            return False
    
    def correct_text(self, text, max_new_tokens=100):
        """
        Qwenを使用したテキスト校正
        
        Args:
            text (str): 校正対象のテキスト
            max_new_tokens (int): 最大生成トークン数
            
        Returns:
            dict: 校正結果
        """
        if not self.model or not self.tokenizer:
            return {
                "original": text,
                "corrected": text,
                "success": False,
                "error": "モデルが利用できません"
            }
        
        start_time = time.time()
        
        try:
            # Qwen用プロンプト作成（Chat形式）
            prompt = f"""以下の日本語文を自然で読みやすい文章に修正してください。誤字脱字の訂正、適切な漢字の使用、自然な表現への改善を行い、修正後の文章のみを出力してください。

修正前: {text}
修正後:"""
            
            # Chat形式でのメッセージ構築
            messages = [
                {
                    "role": "system", 
                    "content": "あなたは日本語の文章校正を専門とするAIアシスタントです。誤字脱字の修正、適切な漢字の使用、自然な表現への改善を行ってください。"
                },
                {
                    "role": "user", 
                    "content": f"次の文章を校正してください：{text}"
                }
            ]
            
            # チャットテンプレート適用
            text_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # トークン化
            inputs = self.tokenizer(
                text_input,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成設定（品質重視）
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.3,  # 創造性を抑えて確実性重視
                "top_p": 0.8,
                "do_sample": True,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # テキスト生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # 結果デコード
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # 入力部分を除去して修正部分のみ抽出
            corrected_text = generated_text.split("assistant")[-1].strip()
            
            # 不要な前置詞や記号を除去
            corrected_text = corrected_text.replace("修正後:", "").strip()
            corrected_text = corrected_text.replace("「", "").replace("」", "")
            corrected_text = corrected_text.split("\n")[0].strip()  # 最初の行のみ
            
            # 元のテキストと同じ場合は変更なしとする
            if corrected_text == text or not corrected_text:
                corrected_text = text
            
            processing_time = time.time() - start_time
            
            return {
                "original": text,
                "corrected": corrected_text,
                "success": True,
                "processing_time": processing_time,
                "changed": corrected_text != text
            }
            
        except Exception as e:
            return {
                "original": text,
                "corrected": text,
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def test_correction(self):
        """校正機能のテスト"""
        
        test_cases = [
            "今日わとてもよい天気ですね。",
            "会議の次弟は明日の午後３時からです。",
            "この商品の個客満足度は高いです。",
            "シュミレーションの結果を確認してください。",
            "お疲れ様でした！！！",
            "こんにちわ、いかがお過ごしですか？？？",
            "システムの　アップデート　が　完了しました",
            "プロジェクトの進捗状況はいかがですか",
            "コミニケーションを取って進めましょう",
            "雰意気がとてもよいですね",
        ]
        
        print("\n📝 Qwen日本語校正テスト")
        print("=" * 60)
        
        total_tests = len(test_cases)
        successful_corrections = 0
        total_time = 0
        
        for i, test_text in enumerate(test_cases, 1):
            print(f"\n--- テスト {i}/{total_tests} ---")
            result = self.correct_text(test_text)
            
            print(f"修正前: 「{result['original']}」")
            print(f"修正後: 「{result['corrected']}」")
            
            if result['success']:
                if result['changed']:
                    print("✅ 修正適用")
                    successful_corrections += 1
                else:
                    print("⚪ 修正不要")
                print(f"処理時間: {result['processing_time']:.2f}秒")
                total_time += result['processing_time']
            else:
                print(f"❌ エラー: {result.get('error', '不明なエラー')}")
        
        # 結果サマリー
        print(f"\n📊 テスト結果サマリー")
        print(f"総テスト数: {total_tests}")
        print(f"修正適用数: {successful_corrections}")
        print(f"修正率: {successful_corrections/total_tests*100:.1f}%")
        print(f"平均処理時間: {total_time/total_tests:.2f}秒")
        
        return successful_corrections, total_tests

def main():
    """メイン実行関数"""
    print("🚀 Qwen2.5-7B-Instruct 日本語校正システムテスト")
    print("=" * 60)
    
    # GPU確認
    if torch.cuda.is_available():
        print(f"✅ GPU利用可能: {torch.cuda.get_device_name(0)}")
        print(f"   GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️  CPU使用（処理時間が長くなる可能性があります）")
    
    try:
        # Qwen校正システム初期化
        corrector = QwenTextCorrector()
        
        if corrector.model and corrector.tokenizer:
            # テスト実行
            successful, total = corrector.test_correction()
            
            print(f"\n✅ テスト完了")
            print(f"🎯 Qwen2.5-7B-Instruct (4bit量子化)")
            print(f"   - 修正成功率: {successful/total*100:.1f}%")
            print(f"   - 4bit量子化によるメモリ効率化")
            print(f"   - 多言語対応・日本語最適化")
            
        else:
            print("❌ モデル初期化に失敗しました")
            
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # メモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()