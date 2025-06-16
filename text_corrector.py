#!/usr/bin/env python3
"""
日本語テキスト校正・誤字訂正システム
軽量かつ高品質な文章改善を実現
"""

import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import jaconv
import time

class JapaneseTextCorrector:
    """日本語テキスト校正システム"""
    
    def __init__(self, use_llm=True, model_name="Qwen/Qwen3-8B"):
        """
        初期化
        
        Args:
            use_llm (bool): LLMを使用するかどうか
            model_name (str): 使用するLLMモデル名（推奨: Qwen/Qwen3-8B）
        """
        self.use_llm = use_llm
        self.model_name = model_name
        self.llm_pipeline = None
        
        # 基本的な誤字訂正辞書
        self.correction_dict = {
            # 助詞の誤用
            "こんにちわ": "こんにちは",
            "こんばんわ": "こんばんは",
            "ありがとうございました": "ありがとうございました",
            
            # よくある誤字
            "雰意気": "雰囲気",
            "シュミレーション": "シミュレーション",
            "コミニケーション": "コミュニケーション",
            "シミレーション": "シミュレーション",
            "アクセサリ": "アクセサリー",
            "バッテリ": "バッテリー",
            
            # 敬語の誤用
            "させていただく": "いたします",
            "お疲れ様でした": "お疲れさまでした",
            
            # 漢字の誤用
            "個客": "顧客",
            "次弟": "次第",
            "同志": "同士",
            "意志": "意思",
        }
        
        # 文字種統一ルール
        self.normalization_rules = [
            # 数字の全角→半角統一
            (r'[０-９]', lambda m: str(ord(m.group()) - ord('０'))),
            # 英字の全角→半角統一
            (r'[Ａ-Ｚａ-ｚ]', lambda m: chr(ord(m.group()) - ord('Ａ') + ord('A')) if 'Ａ' <= m.group() <= 'Ｚ' else chr(ord(m.group()) - ord('ａ') + ord('a'))),
            # 記号の統一
            (r'！', '!'),
            (r'？', '?'),
            (r'．', '.'),
            (r'，', ','),
        ]
        
        if self.use_llm:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """LLMモデルの初期化（Qwen2.5-7B-Instruct 4bit量子化）"""
        try:
            print(f"🔄 LLMモデル初期化中: {self.model_name}")
            start_time = time.time()
            
            # GPU利用可能性確認
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 4bit量子化設定
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # トークナイザー読み込み
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # モデル読み込み（4bit量子化）
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config if device == "cuda" else None,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # パディングトークン設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            load_time = time.time() - start_time
            print(f"✅ LLMモデル初期化完了 ({load_time:.1f}秒)")
            
            # GPU メモリ使用量確認
            if device == "cuda":
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"📊 GPU メモリ使用量: {memory_used:.1f}GB")
            
        except Exception as e:
            print(f"⚠️  LLMモデル初期化失敗: {e}")
            print("📝 基本的な文字列処理のみで動作します")
            self.use_llm = False
            self.model = None
            self.tokenizer = None
    
    def basic_correction(self, text):
        """基本的な誤字訂正と正規化"""
        
        # 文字種正規化
        corrected = text
        for pattern, replacement in self.normalization_rules:
            if callable(replacement):
                corrected = re.sub(pattern, replacement, corrected)
            else:
                corrected = corrected.replace(pattern, replacement)
        
        # ひらがな・カタカナ正規化
        corrected = jaconv.normalize(corrected)
        
        # 辞書ベース誤字訂正
        for wrong, correct in self.correction_dict.items():
            corrected = corrected.replace(wrong, correct)
        
        # 連続する句読点の整理
        corrected = re.sub(r'[、。]{2,}', '。', corrected)
        corrected = re.sub(r'[，．]{2,}', '。', corrected)
        
        # 不要な空白の除去
        corrected = re.sub(r'\s+', ' ', corrected)
        corrected = corrected.strip()
        
        return corrected
    
    def llm_correction(self, text, max_new_tokens=80):
        """Qwen2.5-7B-Instructを使用した高度な誤字訂正"""
        
        if not self.use_llm or not hasattr(self, 'model') or not self.model:
            return text
        
        try:
            # 単純な指示プロンプト（日本語確実出力）
            prompt = f"""以下の日本語文章の誤字脱字を修正して、正しい日本語に校正してください。

元の文章: {text}
校正後:"""
            
            text_input = prompt
            
            # トークン化
            inputs = self.tokenizer(
                text_input,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # GPU設定
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 生成設定（確実性重視・日本語出力）
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.1,  # より低温度で確実な日本語出力
                "top_p": 0.8,
                "do_sample": True,
                "repetition_penalty": 1.1,
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
            
            # 結果デコード（入力部分を除去）
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # 校正結果の抽出と清理
            corrected_text = generated_text.strip()
            
            # <think>タグとその内容を除去
            import re
            corrected_text = re.sub(r'<think>.*?</think>', '', corrected_text, flags=re.DOTALL)
            corrected_text = corrected_text.replace('<think>', '').replace('</think>', '')
            
            # 不要な前置詞や記号を除去
            corrected_text = corrected_text.replace("修正後:", "").strip()
            corrected_text = corrected_text.replace("校正後:", "").strip()
            corrected_text = corrected_text.replace("「", "").replace("」", "")
            corrected_text = corrected_text.split("\n")[0].strip()  # 最初の行のみ
            
            # 空の結果や元テキストと同じ場合
            if not corrected_text or corrected_text == text:
                return text
            
            return corrected_text
                
        except Exception as e:
            print(f"⚠️  LLM訂正エラー: {e}")
            return text
    
    def correct_text(self, text, use_advanced=True):
        """
        総合的なテキスト校正
        
        Args:
            text (str): 校正対象のテキスト
            use_advanced (bool): LLMによる高度な校正を使用するか
            
        Returns:
            dict: 校正結果と情報
        """
        start_time = time.time()
        
        # 入力検証
        if not text or not text.strip():
            return {
                "original": text,
                "corrected": text,
                "changes": [],
                "processing_time": 0,
                "method": "no_processing"
            }
        
        # 基本校正
        basic_corrected = self.basic_correction(text)
        changes = []
        
        if basic_corrected != text:
            changes.append("基本的な誤字訂正・正規化")
        
        # 高度な校正（LLM使用）
        final_corrected = basic_corrected
        if use_advanced and self.use_llm:
            llm_corrected = self.llm_correction(basic_corrected)
            if llm_corrected and llm_corrected != basic_corrected:
                final_corrected = llm_corrected
                changes.append("LLMによる高度な校正")
        
        processing_time = time.time() - start_time
        
        return {
            "original": text,
            "corrected": final_corrected,
            "changes": changes,
            "processing_time": processing_time,
            "method": "llm" if use_advanced and self.use_llm else "basic"
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
        ]
        
        print("📝 テキスト校正機能テスト")
        print("=" * 50)
        
        for i, test_text in enumerate(test_cases, 1):
            print(f"\n--- テスト {i} ---")
            result = self.correct_text(test_text)
            
            print(f"修正前: {result['original']}")
            print(f"修正後: {result['corrected']}")
            print(f"変更点: {', '.join(result['changes']) if result['changes'] else 'なし'}")
            print(f"処理時間: {result['processing_time']:.3f}秒")
            print(f"使用手法: {result['method']}")

def main():
    """メイン実行関数"""
    print("🚀 日本語テキスト校正システム初期化")
    
    # 軽量版で初期化（問題があれば基本版にフォールバック）
    corrector = JapaneseTextCorrector(use_llm=True, model_name="rinna/japanese-gpt-neox-small")
    
    # テスト実行
    corrector.test_correction()
    
    print(f"\n✅ 初期化完了")
    print(f"💡 使用方法:")
    print(f"   corrector = JapaneseTextCorrector()")
    print(f"   result = corrector.correct_text('校正したいテキスト')")

if __name__ == "__main__":
    main()