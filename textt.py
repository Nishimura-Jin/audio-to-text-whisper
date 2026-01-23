import whisper
from datetime import timedelta
from srt import Subtitle
import srt

# modelは使っているグラフィックカードに応じて変更する
model = whisper.load_model("tiny")

# sample.m4aには読み込みたい音声ファイルを指定する
audio = whisper.load_audio(R"C:\Users\lunat\OneDrive\ドキュメント\Python\test3.mp4")

# 再生時間の長い音声ファイルを冒頭30秒だけ試したいときは以下のコードを有効にする
#audio = whisper.pad_or_trim(audio)

mel = whisper.log_mel_spectrogram(audio).to(model.device)
result = model.transcribe(audio, verbose=True, language="ja" )
seginfo = result["segments"]
out_text = []

# segment情報から発言の開始/終了時間とテキストを抜き出し、srt形式で編集する
for data in seginfo:
    start = data["start"]
    end = data["end"]
    text = data["text"]
    out_line = Subtitle(index=1,\
                        start=timedelta(seconds=timedelta(seconds=start).seconds,\
                        microseconds=timedelta(seconds=start).microseconds),\
                        end=timedelta(seconds=timedelta(seconds=end).seconds,\
                        microseconds=timedelta(seconds=end).microseconds),\
                        content=text,\
                        proprietary='')
    out_text.append(out_line)

# srt形式のファイルをcsv形式に編集して保存する。sampleの部分は自由に変更できる
with open("sampl" + ".csv", mode="w", encoding="utf-8") as f:
    origin = srt.compose(out_text)
    origin = origin.replace(",",".")
    origin = origin.replace("\n",",")
    origin = origin.replace(",,","\n")
    f.write(origin)