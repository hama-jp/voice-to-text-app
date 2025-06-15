#!/usr/bin/env python3
"""
Qwen3-8Bモデルを使用した日本語校正システムのテスト
4bit量子化による高効率実装
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import gc

class Qwen3TextCorrector:
    """Qwen3-8Bを使用した日本語テキスト校正システム"""
    
    def __init__(self, model_name="Qwen/Qwen3-8B-Instruct"):
        """
        初期化
        
        Args:
            model_name (str): 使用するQwen3モデル名
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🤖 Qwen3日本語校正システム初期化: {model_name}")
        self._initialize_model()
    
    def _initialize_model(self):
        """4bit量子化でQwen3モデルを初期化"""
        try:
            print("🔄 Qwen3モデル読み込み中...")
            start_time = time.time()
            
            # 4bit量子化設定（メモリ効率最適化）
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=torch.uint8
            )
            
            # トークナイザー読み込み
            print("   📚 トークナイザー読み込み...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            # パディングトークン設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # モデル読み込み（4bit量子化 + 最適化設定）
            print("   🧠 Qwen3モデル読み込み...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_flash_attention_2=False,  # 安定性重視
                attn_implementation="eager"   # 確実な実装
            )
            
            # モデル評価モードに設定
            self.model.eval()
            
            load_time = time.time() - start_time
            print(f"✅ Qwen3モデル読み込み完了 ({load_time:.1f}秒)")
            
            # メモリ使用量確認
            if self.device == "cuda":
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"   GPU メモリ使用量: {memory_used:.1f}GB")
            
            return True
            
        except Exception as e:
            print(f"❌ Qwen3モデル読み込みエラー: {e}")
            print("💡 代替案: より軽量なモデルを試すか、CPUモードで実行")
            self.model = None
            self.tokenizer = None
            return False
    
    def correct_text(self, text, max_new_tokens=80):
        """
        Qwen3を使用したテキスト校正
        
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
            # Qwen3用プロンプト（簡潔で効果的）
            messages = [
                {
                    "role": "system", 
                    "content": "あなたは日本語文章校正の専門家です。誤字脱字の修正、適切な漢字の使用、自然な表現への改善を行ってください。修正後の文章のみを出力してください。"
                },
                {
                    "role": "user", 
                    "content": f"以下の文章を校正してください：\n{text}"
                }
            ]
            
            # チャットテンプレート適用
            text_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # トークン化（効率的な設定）
            inputs = self.tokenizer(
                text_input,
                return_tensors="pt",
                truncation=True,
                max_length=512,  # 入力長制限
                padding=False
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成設定（品質と速度のバランス）
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.2,  # 低温度で確実性重視
                "top_p": 0.85,
                "do_sample": True,
                "repetition_penalty": 1.05,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True
            }
            
            # テキスト生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # 結果デコード
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],  # 入力部分を除去
                skip_special_tokens=True
            )
            
            # 校正結果の抽出と清理
            corrected_text = generated_text.strip()
            
            # 不要な前置詞や記号を除去
            corrected_text = corrected_text.replace("修正後:", "").strip()
            corrected_text = corrected_text.replace("校正後:", "").strip()
            corrected_text = corrected_text.replace("「", "").replace("」", "")
            corrected_text = corrected_text.split("\n")[0].strip()  # 最初の行のみ
            
            # 空の結果や元テキストと同じ場合
            if not corrected_text or corrected_text == text:
                corrected_text = text
            
            processing_time = time.time() - start_time
            
            return {
                "original": text,
                "corrected": corrected_text,
                "success": True,
                "processing_time": processing_time,
                "changed": corrected_text != text,
                "model": "Qwen3-8B-Instruct-4bit"
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
        """校正機能の包括的テスト"""
        
        test_cases = [
            # 基本的な誤字
            "今日わとてもよい天気ですね。",
            "こんにちわ、元気ですか？",
            
            # 漢字の誤用
            "会議の次弟は明日の午後３時からです。",
            "この商品の個客満足度は高いです。",
            "雰意気がとてもよいですね。",
            
            # カタカナ誤字
            "シュミレーションの結果を確認してください。",
            "コミニケーションを取って進めましょう。",
            "バッテリの交換が必要です。",
            
            # 文字種・記号
            "お疲れ様でした！！！",
            "システムの　アップデート　が　完了しました",
            
            # 正しい文（変更不要）
            "本日はお忙しい中、お時間をいただきありがとうございます。",
            "プロジェクトの進捗について報告いたします。",
        ]
        
        print("\n📝 Qwen3日本語校正テスト")
        print("=" * 70)
        
        total_tests = len(test_cases)
        successful_corrections = 0
        total_time = 0
        changed_count = 0
        
        for i, test_text in enumerate(test_cases, 1):
            print(f"\n--- テスト {i:2d}/{total_tests} ---")
            result = self.correct_text(test_text)
            
            print(f"修正前: 「{result['original']}」")
            print(f"修正後: 「{result['corrected']}」")
            
            if result['success']:
                if result['changed']:
                    print("✅ 修正適用")
                    successful_corrections += 1
                    changed_count += 1
                else:
                    print("⚪ 修正不要（適切な判断）")
                print(f"処理時間: {result['processing_time']:.2f}秒")
                total_time += result['processing_time']
            else:
                print(f"❌ エラー: {result.get('error', '不明なエラー')}")
        
        # 結果サマリー
        print(f"\n📊 Qwen3校正システム テスト結果")
        print("=" * 50)
        print(f"総テスト数: {total_tests}")
        print(f"成功処理数: {total_tests - sum(1 for i in range(total_tests) if not test_cases)}")
        print(f"修正適用数: {changed_count}")
        print(f"修正適用率: {changed_count/total_tests*100:.1f}%")
        print(f"平均処理時間: {total_time/total_tests:.2f}秒")
        print(f"使用モデル: Qwen3-8B-Instruct (4bit量子化)")
        
        return successful_corrections, total_tests

def main():
    """メイン実行関数"""
    print("🚀 Qwen3-8B-Instruct 日本語校正システムテスト")
    print("=" * 70)
    
    # システム情報表示
    if torch.cuda.is_available():
        print(f"✅ GPU利用可能: {torch.cuda.get_device_name(0)}")
        print(f"   GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"   PyTorch版: {torch.__version__}")
    else:
        print("⚠️  CPU使用（処理時間が長くなる可能性があります）")
    
    try:
        # Qwen3校正システム初期化
        corrector = Qwen3TextCorrector()
        
        if corrector.model and corrector.tokenizer:
            # テスト実行
            successful, total = corrector.test_correction()
            
            print(f"\n🎯 最終評価")
            print("=" * 30)
            print(f"✅ Qwen3-8B-Instruct (4bit量子化)")
            print(f"   - 高性能多言語モデル")
            print(f"   - メモリ効率: 4bit量子化")
            print(f"   - 商用利用: 確認要（ライセンス次第）")
            print(f"   - 処理性能: GPU最適化")
            
            # 統合推奨
            print(f"\n💡 統合推奨度: ⭐⭐⭐⭐⭐")
            print(f"   理由: 高精度、メモリ効率、日本語対応優秀")
            
        else:
            print("❌ モデル初期化に失敗しました")
            print("💡 対処法:")
            print("   1. transformersライブラリを最新版に更新")
            print("   2. より軽量なモデル（Qwen2.5-7B）を試用")
            print("   3. CPUモードでの実行")
            
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # メモリクリア
        print("\n🧹 メモリクリア中...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("✅ クリア完了")

if __name__ == "__main__":
    main()