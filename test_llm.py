#!/usr/bin/env python3
"""
日本語誤字訂正用LLMモデルの動作検証スクリプト
商用利用可能な高品質モデルの選定とテスト
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time
import gc

def test_gpu_availability():
    """GPU利用可能性とメモリ確認"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✅ CUDA利用可能: {device_count}個のGPU")
        for i in range(device_count):
            properties = torch.cuda.get_device_properties(i)
            total_memory = properties.total_memory / 1024**3
            print(f"   GPU {i}: {properties.name}, {total_memory:.1f}GB")
        return True
    else:
        print("⚠️  GPU利用不可、CPUを使用")
        return False

def test_model_loading(model_name, use_quantization=True):
    """モデル読み込みテスト"""
    print(f"\n🔄 モデル読み込み: {model_name}")
    start_time = time.time()
    
    try:
        # GPU利用可能な場合のデバイス設定
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 量子化設定（GPU利用時のメモリ節約）
        model_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "device_map": "auto" if device == "cuda" else None,
        }
        
        if use_quantization and device == "cuda":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
            print("   📦 4bit量子化を使用（メモリ最適化）")
        
        # トークナイザー読み込み
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # モデル読み込み
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            **model_kwargs
        )
        
        load_time = time.time() - start_time
        print(f"✅ モデル読み込み完了 ({load_time:.1f}秒)")
        
        # メモリ使用量確認
        if device == "cuda":
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"   GPU メモリ使用量: {memory_used:.1f}GB")
        
        return model, tokenizer, device
        
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        return None, None, None

def test_japanese_correction(model, tokenizer, device):
    """日本語誤字訂正テスト"""
    
    if not model or not tokenizer:
        print("❌ モデルまたはトークナイザーが利用できません")
        return
    
    print(f"\n📝 日本語誤字訂正テスト開始")
    
    # テスト用の誤字を含む日本語文
    test_texts = [
        "今日はとてもよい天気ですね。",  # 正しい文（ベースライン）
        "きょうはとてもよいてんきですね。",  # ひらがな過多
        "今日わとてもよい天気ですね。",  # 「は」→「わ」誤字
        "会議の次弟は明日の午後三時からです。",  # 「次第」→「次弟」誤字
        "この商品の個客満足度は高いです。",  # 「顧客」→「個客」誤字
    ]
    
    correction_prompt = """以下の日本語文を自然で読みやすい文章に修正してください。誤字脱字があれば訂正し、より適切な表現があれば改善してください。

修正前: {input_text}
修正後:"""
    
    results = []
    
    for i, text in enumerate(test_texts):
        print(f"\n--- テスト {i+1} ---")
        print(f"入力: {text}")
        
        try:
            # プロンプト作成
            prompt = correction_prompt.format(input_text=text)
            
            # トークン化
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 生成設定（品質重視）
            generation_config = {
                "max_new_tokens": 100,
                "temperature": 0.3,  # 創造性を抑えて確実性重視
                "top_p": 0.9,
                "do_sample": True,
                "repetition_penalty": 1.1,
                "pad_token_id": tokenizer.eos_token_id,
            }
            
            start_time = time.time()
            
            # テキスト生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_config
                )
            
            # 結果デコード
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            correction = generated_text.split("修正後:")[-1].strip()
            
            generation_time = time.time() - start_time
            
            print(f"出力: {correction}")
            print(f"処理時間: {generation_time:.2f}秒")
            
            results.append({
                "input": text,
                "output": correction,
                "time": generation_time
            })
            
        except Exception as e:
            print(f"❌ 処理エラー: {e}")
            results.append({
                "input": text,
                "output": f"エラー: {e}",
                "time": 0
            })
    
    # 結果評価
    print(f"\n📊 テスト結果サマリー")
    avg_time = sum(r["time"] for r in results if r["time"] > 0) / len([r for r in results if r["time"] > 0])
    print(f"平均処理時間: {avg_time:.2f}秒")
    
    successful_tests = len([r for r in results if not r["output"].startswith("エラー")])
    print(f"成功率: {successful_tests}/{len(results)} ({successful_tests/len(results)*100:.1f}%)")
    
    return results

def test_commercial_models():
    """商用利用可能なモデルのテスト"""
    
    # 商用利用可能で日本語対応が良好なモデル
    models_to_test = [
        {
            "name": "microsoft/DialoGPT-medium",
            "description": "Microsoft DialoGPT (MITライセンス)",
            "use_quantization": False  # 小型モデルのため量子化不要
        },
        {
            "name": "rinna/japanese-gpt-neox-3.6b",
            "description": "Rinna日本語GPT (MITライセンス)",
            "use_quantization": True
        },
        {
            "name": "cyberagent/open-calm-7b",
            "description": "CyberAgent OpenCALM (Apache 2.0)",
            "use_quantization": True
        },
    ]
    
    print("🚀 商用利用可能な日本語LLMモデルテスト開始")
    print("=" * 60)
    
    # GPU確認
    gpu_available = test_gpu_availability()
    
    best_model = None
    best_score = 0
    
    for model_info in models_to_test:
        print(f"\n{'='*20} {model_info['description']} {'='*20}")
        
        try:
            # モデル読み込み
            model, tokenizer, device = test_model_loading(
                model_info["name"], 
                use_quantization=model_info["use_quantization"] and gpu_available
            )
            
            if model and tokenizer:
                # 日本語訂正テスト
                results = test_japanese_correction(model, tokenizer, device)
                
                # 簡易スコア計算（成功率 + 処理速度）
                if results:
                    success_rate = len([r for r in results if not r["output"].startswith("エラー")]) / len(results)
                    avg_time = sum(r["time"] for r in results if r["time"] > 0) / max(len([r for r in results if r["time"] > 0]), 1)
                    speed_score = max(0, 1 - avg_time / 10)  # 10秒基準で速度スコア
                    total_score = success_rate * 0.7 + speed_score * 0.3
                    
                    print(f"📊 評価スコア: {total_score:.3f} (成功率: {success_rate:.3f}, 速度: {speed_score:.3f})")
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_model = model_info
                
                # メモリクリア
                del model, tokenizer
                if gpu_available:
                    torch.cuda.empty_cache()
                gc.collect()
                
            else:
                print("❌ モデル読み込み失敗")
                
        except Exception as e:
            print(f"❌ テストエラー: {e}")
    
    # 推奨モデル表示
    if best_model:
        print(f"\n🏆 推奨モデル: {best_model['description']}")
        print(f"   モデル名: {best_model['name']}")
        print(f"   評価スコア: {best_score:.3f}")
    else:
        print("\n❌ 利用可能なモデルが見つかりませんでした")
    
    return best_model

if __name__ == "__main__":
    print("🤖 日本語誤字訂正LLMモデル検証テスト")
    print("=" * 50)
    
    try:
        # 商用利用可能なモデルをテスト
        best_model = test_commercial_models()
        
        if best_model:
            print(f"\n✅ テスト完了")
            print(f"🎯 推奨設定:")
            print(f"   - モデル: {best_model['name']}")
            print(f"   - 量子化: {'有効' if best_model['use_quantization'] else '無効'}")
            print(f"   - 用途: 日本語誤字訂正・文章改善")
        else:
            print(f"\n❌ 適切なモデルが見つかりませんでした")
            print(f"💡 代替案: GPT-2ベースまたはT5ベースのモデルを検討")
        
    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        print(f"💡 対処法: 依存パッケージの確認、メモリ不足の解決")