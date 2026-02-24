import whisper
from whisper.utils import get_writer
import os
import argparse
import torch
import re
from tqdm import tqdm

def clean_text(text):
    """
    フィラー（えー、あのー等）を削除し、文章を整える関数
    """
    # 削除対象のリスト（正規表現を活用）
    fillers = [
        r"えーと、?", r"えー、?", r"あのー、?", r"あの、?", 
        r"えっと、?", r"まー、?", r"そのー、?", r"えー"
    ]
    cleaned = text
    for f in fillers:
        cleaned = re.sub(f, "", cleaned)
    
    # 連続する空白を1つにまとめ、前後の不要な空白を削除
    cleaned = cleaned.replace("  ", " ").strip()
    return cleaned

def main():
    # 1. コマンドライン引数の設定
    # これにより「python app.py ファイル名 --format 形式」という操作が可能になります
    parser = argparse.ArgumentParser(description="Whisperを用いた高精度・クレンジング機能付き文字起こしツール")
    parser.add_argument("input_file", help="文字起こししたい音声・動画ファイルのパス")
    parser.add_argument(
        "--format", 
        choices=["all", "srt", "vtt", "txt", "tsv", "json"], 
        default="srt", 
        help="出力形式 (デフォルト: srt)"
    )
    parser.add_argument(
        "--clean", 
        action="store_true", 
        help="フィラー（えー、あの等）を除去して保存する"
    )
    args = parser.parse_args()

    # 2. デバイスの自動判定とモデルロード
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- 起動環境: {device} ---")
    print("--- モデル読み込み中 (turbo) ---")
    model = whisper.load_model("turbo", device=device)

    # 3. 文字起こしの実行
    print(f"--- 処理開始: {os.path.basename(args.input_file)} ---")
    result = model.transcribe(
        args.input_file,
        verbose=True,
        language="ja",
        condition_on_previous_text=False # 無限ループ防止
    )

    # 4. フィラー除去の適用（オプション）
    if args.clean:
        print("✨ フィラー除去を実行中...")
        # 各セグメント（時間ごとの文章）を掃除
        for segment in result["segments"]:
            segment["text"] = clean_text(segment["text"])
        # 全体テキストも掃除
        result["text"] = clean_text(result["text"])

    # 5. 指定されたフォーマットで書き出し
    output_dir = os.path.dirname(args.input_file) or "."
    formats = ["srt", "vtt", "txt", "tsv", "json"] if args.format == "all" else [args.format]

    print(f"💾 {len(formats)}件のファイルを保存しています...")
    for f in tqdm(formats, desc="Saving"):
        writer = get_writer(f, output_dir)
        writer(result, args.input_file)

    print(f"--- すべての処理が完了しました！ ---")

if __name__ == "__main__":
    main()
