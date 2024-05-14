# LLaVA-Python-Sample

Vision and Language Model（画像と自然言語を組み合わせたモデル）に画像の説明をさせるサンプルコード

## Quick Start

初回実行時

```bash
$ make setup
```

2 回目以降

```bash
$ docker-compose up -d
```

open http://localhost:8501

![](./image/demo.png)

画像は https://huggingface.co/rinna/bilingual-gpt-neox-4b-minigpt4/resolve/main/sample.jpg より

## Usage

1. 画像をアップロードする
1. チャット欄にモデルへのプロンプトを記述する

![](./image/top.png)

## Used Model

https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#multi-modal-models
にあるようにいくつかのモデルから好みのものを選択して使うことができる。

ページ上部のモデル一覧から使用したいモデルを選択して実行することができる。

※13B 以上のモデルは GPU 利用を推奨
