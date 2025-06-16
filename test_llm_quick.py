#!/usr/bin/env python3
from text_corrector import JapaneseTextCorrector

print('LLM校正テスト開始...')
corrector = JapaneseTextCorrector(use_llm=True, model_name='Qwen/Qwen3-8B')

test_text = '会議の次弟は明日の午後３時からです。'
result = corrector.correct_text(test_text, use_advanced=True)
print(f'元の文: {result["original"]}')
print(f'校正後: {result["corrected"]}')
print(f'変更あり: {result["corrected"] != result["original"]}')
print(f'処理時間: {result["processing_time"]:.2f}秒')