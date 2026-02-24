import whisper
from whisper.utils import get_writer
import os
import sys
import torch

def main():
    # 1. デバイスの自動検知
    # GPU (CUDA) があれば使用し、なければ CPU で動作させるハイブリッド仕様
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- 起動中 (Device: {device}) ---")

    # 2. 実行時引数のチェック
    # コマンドラインから 'python app.py movie.mp4' のように実行可能
    if len(sys.argv) < 2:
        print("エラー: 入力ファイルが指定されていません。")
        print("使用法: python app.py [ファイルパス]")
        return
    
    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"エラー: ファイルが見つかりません: {audio_path}")
        return

    # 3. モデルのロード
    # 速度と精度のバランスが最も優れた "turbo" モデルを採用
    print(f"--- モデル読み込み中 (Model: turbo) ---")
    model = whisper.load_model("turbo", device=device)

    # 4. 文字起こしの実行
    # 楽曲の認識率向上と、実務での安定性（ハルシネーション対策）を両立した設定
    print(f"--- 処理開始: {os.path.basename(audio_path)} ---")
    result = model.transcribe(
        audio_path,
        verbose=True,           # 進捗をリアルタイム表示（UX向上）
        language="ja",          # 日本語に固定して誤認識を防止
        beam_size=5,            # 探索を深め、歌詞や複雑な文章の精度を向上
        temperature=0,          # 決定論的な出力を優先（変な「遊び」を抑制）
        no_speech_threshold=0.6, # 無音部分で勝手に喋り出す現象を抑制
        condition_on_previous_text=False, # 同じフレーズを繰り返す無限ループを防止
        initial_prompt="これは正確な日本語の字幕用データです。" # 漢字変換の精度を高めるヒント
    )

    # 5. 公式ツールによるSRTファイルの保存
    # ライブラリ依存を減らし、Whisper標準の出力機能を使用して正確なSRTを生成
    output_dir = os.path.dirname(audio_path) if os.path.dirname(audio_path) else "."
    
    # ファイル名（拡張子なし）を取得
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # 字幕書き出し用のWriterを準備
    writer = get_writer("srt", output_dir)
    
    # 実行（resultデータをSRT形式に整形してファイル保存）
    writer(result, audio_path)

    print(f"--- 完了 ---")
    print(f"保存先: {os.path.join(output_dir, base_name + '.srt')}")

if __name__ == "__main__":
    main()


