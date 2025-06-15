#!/usr/bin/env python3
"""
利用可能なQwenモデルの確認と最適モデル選択
"""

import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time

def check_available_qwen_models():
    """利用可能なQwenモデルを確認"""
    
    print("🔍 利用可能なQwenモデル確認中...")
    
    # 推奨順に確認するモデルリスト
    candidate_models = [
        "Qwen/Qwen2.5-7B-Instruct",      # 最新の安定版
        "Qwen/Qwen2-7B-Instruct",        # 前世代の安定版
        "Qwen/Qwen1.5-7B-Chat",          # 軽量版
        "Qwen/Qwen-7B-Chat",             # オリジナル版
        "Qwen/CodeQwen1.5-7B-Chat",      # コード特化（文章も対応）
    ]
    
    available_models = []
    
    for model_name in candidate_models:
        print(f"\n🔄 確認中: {model_name}")
        
        try:
            # ヘッドリクエストでモデル存在確認
            url = f"https://huggingface.co/{model_name}"
            response = requests.head(url, timeout=10)
            
            if response.status_code == 200:
                print(f"  ✅ 利用可能")
                
                # 簡易的なトークナイザー読み込みテスト
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name, 
                        trust_remote_code=True
                    )
                    print(f"  ✅ トークナイザー読み込み成功")
                    available_models.append({
                        "name": model_name,
                        "status": "ready",
                        "description": get_model_description(model_name)
                    })
                    del tokenizer  # メモリ節約
                except Exception as e:
                    print(f"  ⚠️  トークナイザー読み込み失敗: {e}")
                    available_models.append({
                        "name": model_name,
                        "status": "tokenizer_issue",
                        "description": get_model_description(model_name)
                    })
            else:
                print(f"  ❌ アクセス不可 (HTTP {response.status_code})")
                
        except Exception as e:
            print(f"  ❌ 確認エラー: {e}")
    
    return available_models

def get_model_description(model_name):
    """モデルの説明を取得"""
    descriptions = {
        "Qwen/Qwen2.5-7B-Instruct": "最新版・日本語対応強化・商用利用可能",
        "Qwen/Qwen2-7B-Instruct": "安定版・多言語対応・商用利用可能", 
        "Qwen/Qwen1.5-7B-Chat": "軽量版・チャット最適化",
        "Qwen/Qwen-7B-Chat": "オリジナル版・実績豊富",
        "Qwen/CodeQwen1.5-7B-Chat": "コード特化・文章校正も対応",
    }
    return descriptions.get(model_name, "詳細不明")

def test_qwen_model(model_name):
    """選択されたQwenモデルのテスト"""
    
    print(f"\n🧪 {model_name} の動作テスト")
    print("=" * 60)
    
    try:
        start_time = time.time()
        
        # 4bit量子化設定
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        print("🔄 モデル読み込み中...")
        
        # トークナイザー
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # モデル（4bit量子化）
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        load_time = time.time() - start_time
        print(f"✅ モデル読み込み完了 ({load_time:.1f}秒)")
        
        # GPU メモリ使用量確認
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"📊 GPU メモリ使用量: {memory_used:.1f}GB")
        
        # 簡単な校正テスト
        test_text = "今日わとてもよい天気ですね。"
        
        messages = [
            {"role": "system", "content": "日本語文章を校正してください。"},
            {"role": "user", "content": f"次の文章を校正してください：{test_text}"}
        ]
        
        # チャットテンプレート適用
        if hasattr(tokenizer, 'apply_chat_template'):
            text_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # フォールバック
            text_input = f"ユーザー: {test_text}\nアシスタント:"
        
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # 生成テスト
        print("🎯 校正テスト実行中...")
        generation_start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generation_time = time.time() - generation_start
        
        # 結果デコード
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✅ 校正テスト完了 ({generation_time:.2f}秒)")
        print(f"📝 入力: {test_text}")
        print(f"📝 出力: {generated_text[-100:]}")  # 最後の100文字
        
        # クリーンアップ
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        return False

def main():
    """メイン実行"""
    print("🚀 Qwenモデル確認・選択システム")
    print("=" * 50)
    
    # GPU確認
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️  CPU使用")
    
    # 利用可能モデル確認
    available_models = check_available_qwen_models()
    
    if not available_models:
        print("❌ 利用可能なQwenモデルが見つかりません")
        return
    
    print(f"\n📋 利用可能なQwenモデル ({len(available_models)}個)")
    print("=" * 60)
    
    for i, model_info in enumerate(available_models, 1):
        status_icon = "✅" if model_info["status"] == "ready" else "⚠️"
        print(f"{i}. {status_icon} {model_info['name']}")
        print(f"   {model_info['description']}")
    
    # 最適モデル推奨
    if available_models:
        recommended = available_models[0]  # 最初のモデル（優先度順）
        print(f"\n🎯 推奨モデル: {recommended['name']}")
        print(f"   理由: {recommended['description']}")
        
        # 推奨モデルのテスト
        print(f"\n🧪 推奨モデルのテスト実行...")
        success = test_qwen_model(recommended['name'])
        
        if success:
            print(f"\n✅ 統合推奨")
            print(f"🎯 text_corrector.py で使用するモデル:")
            print(f"   model_name = '{recommended['name']}'")
        else:
            print(f"\n⚠️  代替モデルの検討が必要")

if __name__ == "__main__":
    main()