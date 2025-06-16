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
        """Qwen3-8Bを使用した高度な誤字訂正（長文対応）"""
        
        if not self.use_llm or not hasattr(self, 'model') or not self.model:
            return text
        
        # 長文の場合は分割処理（VRAM 11.5GB環境向け調整）
        if len(text) > 6000:  # 6000文字以上は分割（さらに拡張）
            return self._process_long_text(text, max_new_tokens)
        
        try:
            # 具体的で明確な校正指示プロンプト
            prompt = f"""以下の文章の誤字脱字を修正してください。説明不要。修正した文章のみ出力。

{text}"""
            
            text_input = prompt
            
            # メモリクリア（VRAM最適化）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # トークン化（VRAM 11.5GB最大活用）
            inputs = self.tokenizer(
                text_input,
                return_tensors="pt",
                truncation=True,
                max_length=8192  # 4096から8192に大幅拡張
            )
            
            # GPU設定
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 生成設定（長文対応・VRAM最適化・トークン大幅拡張）
            generation_config = {
                "max_new_tokens": max(max_new_tokens * 4, len(text) * 2, 2048),  # さらに大幅増加
                "temperature": 0.1,  # より低温度で確実な日本語出力
                "top_p": 0.8,
                "do_sample": True,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": False  # メモリ節約のためキャッシュ無効化
            }
            
            # テキスト生成（メモリ効率化）
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
                
                # 即座にメモリ解放
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 結果デコード（入力部分を除去）
            input_length = len(self.tokenizer.encode(text_input))
            generated_text = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
            
            # 校正結果の抽出と清理
            corrected_text = generated_text.strip()
            
            # 🔍 デバッグ: LLMの生出力を確認
            print(f"🔍 LLM生出力 ({len(generated_text)}文字):")
            print(f"   前半100文字: {generated_text[:100]}")
            print(f"   後半100文字: {generated_text[-100:]}")
            
            # <think>タグとその内容を除去
            import re
            corrected_text = re.sub(r'<think>.*?</think>', '', corrected_text, flags=re.DOTALL)
            corrected_text = corrected_text.replace('<think>', '').replace('</think>', '')
            
            # 修正後の部分のみを抽出
            # パターン1: "修正後:" や "校正後の文章:" などの後の部分を取得
            import re
            patterns = [
                r"修正版:?\s*(.*)",
                r"校正版:?\s*(.*)",
                r"修正後の?文章?:?\s*(.*)",
                r"校正後の?文章?:?\s*(.*)", 
                r"修正済みテキスト:?\s*(.*)",
                r"校正結果:?\s*(.*)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, corrected_text, re.DOTALL)
                if match:
                    corrected_text = match.group(1).strip()
                    break
            
            # 「元の文章」部分を除去（もし含まれていたら）
            if "元の文章:" in corrected_text:
                parts = corrected_text.split("元の文章:")
                if len(parts) > 1:
                    # "元の文章:"より前の部分を取得
                    corrected_text = parts[0].strip()
            
            # 不要な前置詞や記号を除去
            prefixes_to_remove = [
                "修正後:", "校正後:", "校正結果:", "出力:", "校正後の文章:", "修正済みテキスト:",
                "以下は、原文の誤字脱字や文脈の整合性を考慮しながら校正した結果です：",
                "以下が校正結果です：", "校正版：", "修正版：",
                "正しい文章：", "改善された文章：", "訂正後："
            ]
            
            for prefix in prefixes_to_remove:
                corrected_text = corrected_text.replace(prefix, "").strip()
            
            # 引用符除去
            corrected_text = corrected_text.replace("「", "").replace("」", "")
            corrected_text = corrected_text.replace(""", "").replace(""", "")
            
            # 冒頭の余計な文字を除去（「は校正後の...」パターン）
            import re
            # 冒頭に「は」で始まる説明文がある場合、実際のテキストを抽出
            corrected_text = re.sub(r'^は[^。]*。?\s*', '', corrected_text, flags=re.MULTILINE)
            
            # 一般的な説明文を除去
            generic_responses = [
                "は、原意を忠実に反映しながら、読みやすく理解しやすい日本語に仕上げられています。",
                "何か追加のご要望や調整が必要な場合は、遠慮なくお知らせください",
                "校正済みの文章",
                "修正済みの文章",
                "より自然で読みやすい日本語に校正しました",
                "以下が校正した結果です",
                "校正後のテキストのみ出力してほしい"
            ]
            
            for generic in generic_responses:
                if generic in corrected_text:
                    corrected_text = corrected_text.replace(generic, "").strip()
            
            # 有効な内容を抽出（複数行対応）
            lines = corrected_text.split("\n")
            valid_lines = []
            for line in lines:
                line = line.strip()
                if (line and 
                    not line.startswith("以下") and 
                    not line.endswith("：") and 
                    not line.startswith("何か") and
                    not line.startswith("追加") and
                    line not in ["校正済みの文章", "修正済みの文章"]):
                    valid_lines.append(line)
            
            if valid_lines:
                corrected_text = "\n".join(valid_lines)
            else:
                corrected_text = corrected_text.strip()
            
            # 空の結果や元テキストと同じ場合
            if not corrected_text or corrected_text == text:
                return text
            
            return corrected_text
                
        except Exception as e:
            print(f"⚠️  LLM訂正エラー: {e}")
            return text
    
    def _process_long_text(self, text, max_new_tokens=80):
        """長文を分割してLLM校正処理"""
        
        # 文単位で分割（簡易版）
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in "。！？":
                sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        print(f"📝 長文分割処理: {len(text)}文字 → {len(sentences)}文に分割")
        
        # VRAM使用量確認
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"💾 現在のVRAM使用量: {current_memory:.1f}GB")
        
        # 複数文をまとめて処理（VRAM活用・メモリ効率化）
        corrected_sentences = []
        batch_size = 3  # メモリ節約のため3文に削減
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            batch_text = "".join(batch)
            
            if len(batch_text) > 20:  # 短すぎるバッチは処理スキップ
                corrected_batch = self._correct_single_sentence(batch_text, max_new_tokens)
                corrected_sentences.append(corrected_batch)
                print(f"   バッチ{i//batch_size+1}: {len(batch_text)}→{len(corrected_batch)}文字 ({len(batch)}文)")
                
                # バッチ処理後にメモリ解放
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                corrected_sentences.append(batch_text)
        
        return "".join(corrected_sentences)
    
    def _correct_single_sentence(self, sentence, max_new_tokens=80):
        """複数文バッチのLLM校正（VRAM最適化）"""
        
        try:
            prompt = f"""以下の文章の誤字脱字を修正してください。説明不要。修正した文章のみ出力。

{sentence}"""
            
            # メモリクリア（バッチ処理用）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096  # バッチ処理用に大幅拡張
            )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            generation_config = {
                "max_new_tokens": max(max_new_tokens * 3, len(sentence) * 2, 1024),  # 大幅拡張
                "temperature": 0.1,
                "top_p": 0.8,
                "do_sample": True,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": False  # メモリ節約
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)
                
                # メモリ解放（バッチ処理）
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            input_length = len(self.tokenizer.encode(prompt))
            generated_text = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
            
            # 結果清理（バッチ処理版）
            import re
            corrected_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL)
            corrected_text = corrected_text.replace('<think>', '').replace('</think>', '')
            
            # 修正後の部分のみを抽出（バッチ処理版）
            import re
            patterns = [
                r"修正版:?\s*(.*)",
                r"校正版:?\s*(.*)",
                r"修正後の?文章?:?\s*(.*)",
                r"校正後の?文章?:?\s*(.*)", 
                r"修正済みテキスト:?\s*(.*)",
                r"校正結果:?\s*(.*)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, corrected_text, re.DOTALL)
                if match:
                    corrected_text = match.group(1).strip()
                    break
            
            # 「元の文章」部分を除去（バッチ処理版）
            if "元の文章:" in corrected_text:
                parts = corrected_text.split("元の文章:")
                if len(parts) > 1:
                    corrected_text = parts[0].strip()
            
            # 前置詞除去
            prefixes_to_remove = [
                "修正後:", "校正後:", "校正結果:", "出力:", "校正後の文章:", "修正済みテキスト:",
                "以下は、原文の誤字脱字や文脈の整合性を考慮しながら校正した結果です：",
                "以下が校正結果です：", "校正版：", "修正版：",
                "正しい文章：", "改善された文章：", "訂正後："
            ]
            
            for prefix in prefixes_to_remove:
                corrected_text = corrected_text.replace(prefix, "").strip()
            
            # 引用符除去と有効行抽出
            corrected_text = corrected_text.replace("「", "").replace("」", "")
            corrected_text = corrected_text.replace(""", "").replace(""", "")
            
            # 冒頭の余計な文字を除去（バッチ処理版）
            import re
            corrected_text = re.sub(r'^は[^。]*。?\s*', '', corrected_text, flags=re.MULTILINE)
            
            # 一般的な説明文を除去（バッチ処理版）
            generic_responses = [
                "は、原意を忠実に反映しながら、読みやすく理解しやすい日本語に仕上げられています。",
                "何か追加のご要望や調整が必要な場合は、遠慮なくお知らせください",
                "校正済みの文章",
                "修正済みの文章",
                "より自然で読みやすい日本語に校正しました",
                "以下が校正した結果です",
                "校正後のテキストのみ出力してほしい"
            ]
            
            for generic in generic_responses:
                if generic in corrected_text:
                    corrected_text = corrected_text.replace(generic, "").strip()
            
            lines = corrected_text.split("\n")
            valid_lines = []
            for line in lines:
                line = line.strip()
                if (line and 
                    not line.startswith("以下") and 
                    not line.endswith("：") and 
                    not line.startswith("何か") and
                    not line.startswith("追加") and
                    line not in ["校正済みの文章", "修正済みの文章"]):
                    valid_lines.append(line)
            
            if valid_lines:
                corrected_text = "\n".join(valid_lines)
            else:
                corrected_text = corrected_text.strip()
            
            return corrected_text if corrected_text else sentence
            
        except Exception as e:
            print(f"⚠️  文校正エラー: {e}")
            return sentence
    
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
            print(f"🤖 LLM校正開始: {self.model_name}")
            llm_corrected = self.llm_correction(basic_corrected)
            print(f"🤖 LLM校正完了: 元({len(basic_corrected)}文字) → 校正後({len(llm_corrected) if llm_corrected else 0}文字)")
            if llm_corrected and llm_corrected != basic_corrected:
                final_corrected = llm_corrected
                changes.append("LLMによる高度な校正")
                print(f"✅ LLM校正適用済み")
            else:
                print(f"⚪ LLM校正変更なし")
        elif use_advanced:
            print(f"⚠️  LLM校正要求されましたが利用不可: use_llm={self.use_llm}")
        
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