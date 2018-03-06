# sakamotsuリコメンド
---

1〜4, 6はjupyter notebookを使います。

## [1. 画像取得](https//:github.com/mori5602/sakamotsu/01_downloadPictureFromYahoo.ipynb)
## [2. 画像の解像度統一](https//:github.com/mori5602/sakamotsu/02_reSizePicture.ipynb)
## [3. 画像ファイルの増幅](https//:github.com/mori5602/sakamotsu/03_duplicatePictureFile.ipynb)
## [4. 学習スタート](https//:github.com/mori5602/sakamotsu/04_startTrainingSakamotsu.ipynb)
## 5. 学習を長時間繰り返す  
人工知能用のサーバにsshで接続後、screenを使って学習を繰り返し続けて放置します。
```
screen
python 04_startTrainingSakamotsu.py
Ctrl-a d # Ctrlを押しながらa、その後d(Ctrlは離して)
```

様子を見るときは、以下の通り
```
screen -ls #セッションを確認する。
screen -r <セッション番号>
```

## 6. AIの学習結果を試してみる

## 引用
[Chainerのモデルのセーブとロード - 無限グミ](http://toua20001.hatenablog.com/entry/2016/11/15/203332)


