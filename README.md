# Whisper文字起こしツール

OpenAIのWhisperモデルを使った音声→テキスト変換ツールです。

## 機能
- MP4/WAVなどの音声・動画ファイルを入力
- 日本語・英語など多言語対応の文字起こし
- SRT字幕出力可能

## 必要環境
- Python 3.8以上
- GPUがあれば高速（CPUでも可）

## 使い方（基本）

1.スクリプト内の音声ファイルパスを変更
audio = whisper.load_audio(r"C:\path\to\your\audio.mp4")  # ← ここを自分のファイルに

2.実行
whisper_srt.py→ 同じフォルダに sample.srt（または指定した名前）のSRTファイルが出力されます。

## カスタマイズ例

・モデルサイズを変更（精度↑、速度↓）
  model = whisper.load_model("large-v3")  # medium → large-v3 / turbo など
  
・出力ファイル名をカスタム
　with open("sample.srt", mode="w", encoding="utf-8") as f: ←sampleの部分を出力したい名前に変更可能

・英語音声に変更（日本語以外で使いたい場合）
  result = model.transcribe(audio, verbose=True, language="en")
