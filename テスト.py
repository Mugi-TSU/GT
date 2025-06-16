import re
import csv
from janome.tokenizer import Tokenizer
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import chardet  # chardetをインポート


# 形態素解析器の初期化
tokenizer = Tokenizer()


def analyze_sentence(sentence):
    # 文の文字数
    character_count = len(sentence)

    # 文の形態素解析
    tokens = tokenizer.tokenize(sentence)

    # 単語数と単語の長さ
    words = [token.surface for token in tokens]
    word_count = len(words)
    word_lengths = [len(word) for word in words]

    # 単語の平均長
    average_word_length = sum(word_lengths) / word_count if word_count > 0 else 0

    return {
        "character_count": character_count,
        "word_count": word_count,
        "average_word_length": average_word_length
    }


# ファイルのエンコーディングを検出する
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']


# GUIでファイルを選択する
def select_file():
    # Tkinterの初期化
    root = Tk()
    root.withdraw()  # メインウィンドウを非表示にする

    # ファイル選択ダイアログを開く
    file_path = askopenfilename(filetypes=[("Text files", "*.txt")])
    return file_path

# CSVファイルに書き込む
def write_to_csv(results, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # ヘッダー行の書き込み
        writer.writerow(['文番号', '文字数', '単語数', '平均単語長'])
        # 各文のデータを書き込み
        for i, result in enumerate(results):
            writer.writerow([i+1, result['character_count'], result['word_count'], result['average_word_length']])

# メイン関数
def main():
    # ファイルを選択
    file_path = select_file()

    # ファイルが選択されていない場合
    if not file_path:
        print("ファイルが選択されませんでした。")
        return

    # エンコーディングの検出
    encoding = detect_encoding(file_path)

    try:
        # 選択されたファイルからテキストを指定されたエンコーディングで読み込む
        with open(file_path, 'r', encoding=encoding) as file:
            text = file.read()
    except UnicodeDecodeError as e:
        print(f"ファイルの読み込み中にエラーが発生しました: {e}")
        return

    # 正規表現を使って文を分割する
    sentences = re.split(r'(?<=[。！？])', text)

    # 各文の分析結果を格納するリスト
    results = []

    # 各文に対して分析を実行
    for i, sentence in enumerate(sentences):
        if sentence.strip():  # 空白行や空の文をスキップ
            result = analyze_sentence(sentence)
            results.append(result)
            print(f"文{i + 1}:")
            print(f"  文字数: {result['character_count']}")
            print(f"  単語数: {result['word_count']}")
            print(f"  平均単語長: {result['average_word_length']:.2f}")
            print()

# CSVファイルに書き込む
    output_file = 'sentence_analysis.csv'
    write_to_csv(results, output_file)
    print(f"分析結果を '{output_file}' に書き込みました。")

if __name__ == "__main__":
    main()
