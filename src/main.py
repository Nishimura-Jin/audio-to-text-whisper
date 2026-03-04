import whisper
from whisper.utils import get_writer
import os
import argparse
import torch
import re
from tqdm import tqdm


def clean_text(text):
    # フィラー（えー、あのー等）を削除する
    fillers = [
        r"えーと、?", r"えー、?", r"あのー、?", r"あの、?",
        r"えっと、?", r"まー、?", r"そのー、?", r"えー"
    ]
    cleaned = text
    for f in fillers:
        cleaned = re.sub(f, "", cleaned)

    # 連続する空白を1つにまとめる
    cleaned = cleaned.replace("  ", " ").strip()
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Whisperを使った文字起こし・フィラー除去ツール")
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"起動環境: {device}")
    print("モデル読み込み中 (turbo)...")
    model = whisper.load_model("turbo", device=device)

    print(f"処理開始: {os.path.basename(args.input_file)}")
    result = model.transcribe(
        args.input_file,
        verbose=True,
        language="ja",
        condition_on_previous_text=False  # 誤認の連鎖を防ぐ
    )

    if args.clean:
        print("フィラー除去を実行中...")
        for segment in result["segments"]:
            segment["text"] = clean_text(segment["text"])
        result["text"] = clean_text(result["text"])

    output_dir = os.path.dirname(args.input_file) or "."
    formats = ["srt", "vtt", "txt", "tsv", "json"] if args.format == "all" else [args.format]

    print(f"{len(formats)}件のファイルを保存中...")
    for f in tqdm(formats, desc="Saving"):
        writer = get_writer(f, output_dir)
        writer(result, args.input_file)

    print("完了しました")


if __name__ == "__main__":
    main()
