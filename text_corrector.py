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
    
    def __init__(self):
        """
        初期化
        """
        self.models = {}
        self.tokenizers = {}
        
        # 音声認識誤認識修正辞書（技術的エラーのみ）
        self.correction_dict = {
            # 助詞の音声認識誤認識
            "こんにちわ": "こんにちは",
            "こんばんわ": "こんばんは",
            "きょうわ": "今日は",
            "あしたわ": "明日は",
            "それわ": "それは",
            "これわ": "これは",
            "あれわ": "あれは",
            "どれわ": "どれは",
            "だれわ": "誰は",
            "なにわ": "何は",
            "いつわ": "いつは",
            "どこわ": "どこは",
            "わたしわ": "私は",
            "あなたわ": "あなたは",
            "かれわ": "彼は",
            "かのじょわ": "彼女は",
            
            # 同音異義語の変換ミス（音声認識特有）
            "個客": "顧客",
            "次弟": "次第",
            "同志": "同士",
            "雰意気": "雰囲気",
            "シュミレーション": "シミュレーション",
            "コミニケーション": "コミュニケーション",
            "シミレーション": "シミュレーション",
            
            # カタカナ語の長音記号省略（音声認識でよく発生）
            "アクセサリ": "アクセサリー",
            "バッテリ": "バッテリー",
            "コンピュータ": "コンピューター",
            "プリンタ": "プリンター",
            "スキャナ": "スキャナー",
            "モニタ": "モニター",
            "ユーザ": "ユーザー",
            "サーバ": "サーバー",
            "ブラウザ": "ブラウザー",
            "フォルダ": "フォルダー",
            
            # 数字の全角→半角統一（音声認識で全角になりがち）
            "１": "1", "２": "2", "３": "3", "４": "4", "５": "5",
            "６": "6", "７": "7", "８": "8", "９": "9", "０": "0",
            
            # 音声認識でひらがな化された一般的な単語
            "きょう": "今日",
            "あした": "明日",
            "きのう": "昨日",
            "てんき": "天気",
            "じかん": "時間",
            "ばしょ": "場所",
            "かいしゃ": "会社",
            "しごと": "仕事",
            "がっこう": "学校",
            "でんわ": "電話",
            "ほん": "本",
            "てれび": "テレビ",
            "ぱそこん": "パソコン",
            "けいたい": "携帯",
            
            # 送り仮名の統一（音声認識で送り仮名が付きがち）
            "売り上げ": "売上",
            "取り引き": "取引",
            "申し込み": "申込",
            "打ち合わせ": "打合せ",
            "見積もり": "見積り",
            "手続き": "手続",
            "受け付け": "受付",
            "振り込み": "振込",
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
    
    def load_model(self, model_name):
        """LLMモデルの動的読み込み"""
        if model_name in self.models:
            return

        try:
            print(f"🔄 LLMモデル初期化中: {model_name}")
            start_time = time.time()
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            quantization_config = None
            if "Qwen" in model_name and device == "cuda":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            load_time = time.time() - start_time
            print(f"✅ LLMモデル初期化完了 ({load_time:.1f}秒): {model_name}")
            
            if device == "cuda":
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"📊 GPU メモリ使用量: {memory_used:.1f}GB")
            
        except Exception as e:
            print(f"⚠️  LLMモデル初期化失敗: {e}")
            if model_name in self.models:
                del self.models[model_name]
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]

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
    
    def llm_correction(self, text, model_name, max_new_tokens=80):
        """指定されたLLMを使用した高度な誤字訂正"""
        
        if model_name not in self.models:
            self.load_model(model_name)
            if model_name not in self.models:
                print(f"⚠️  LLMモデルが利用できないため、校正をスキップします: {model_name}")
                return text

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        try:
            # 高精度日本語校正プロンプト
            prompt = f'''あなたは日本語校正の専門家です。音声認識で生成されたテキストを、自然で正確な日本語に校正してください。

【校正項目】
1. 誤字脱字の修正（同音異義語、変換ミス）
2. 助詞の適切な使い分け（は/が、を/に、で/と等）
3. 敬語・丁寧語の自然な表現
4. 文章構造の改善（主語述語の対応、修飾関係）
5. 読点・句点の適切な配置

【厳格なルール】
- 元の文章の意味・意図を絶対に変更しない
- 事実・数値・固有名詞は原文通り保持
- 音声認識特有の誤認識パターンを考慮する
- 口語的表現は適度に書き言葉に調整
- 校正結果のみを簡潔に出力する

【音声認識誤認識の例】
入力: きょうわとてもいいてんきですね
校正: 今日はとても良い天気ですね

入力: かいぎのしだいわあしたのごごさんじからです
校正: 会議の次第は明日の午後３時からです

入力: このしょうひんのこきゃくまんぞくどわたかいです
校正: この商品の顧客満足度は高いです

【校正対象】
入力: {text}
校正:'''
            
            text_input = prompt
            
            # トークン化
            inputs = tokenizer(
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
                "temperature": 0.1,
                "top_p": 0.8,
                "do_sample": True,
                "repetition_penalty": 1.1,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True
            }
            
            # テキスト生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_config
                )
            
            # 結果デコード（入力部分を除去）
            generated_text = tokenizer.decode(
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
    
    def correct_text(self, text, model_name="rinna/japanese-gpt-neox-small"):
        """
        総合的なテキスト校正
        
        Args:
            text (str): 校正対象のテキスト
            model_name (str): 使用するLLMのモデル名
            
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
        if model_name:
            llm_corrected = self.llm_correction(basic_corrected, model_name)
            if llm_corrected and llm_corrected != basic_corrected:
                final_corrected = llm_corrected
                changes.append(f"LLMによる高度な校正 ({model_name})")
        
        processing_time = time.time() - start_time
        
        return {
            "original": text,
            "corrected": final_corrected,
            "changes": changes,
            "processing_time": processing_time,
            "method": f"llm ({model_name})" if model_name else "basic"
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