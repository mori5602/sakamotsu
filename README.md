# sakamotsuリコメンド
---

## プレゼン資料
[第一回人工知能勉強会](https://gitpitch.com/mori5602/sakamotsu)

1〜4, 6はjupyter notebookを使います。

## [1. 画像取得](https://github.com/mori5602/sakamotsu/blob/master/01_downloadPictureFromYahoo.ipynb)
## [2. 画像の解像度統一](https://github.com/mori5602/sakamotsu/blob/master/02_reSizePicture.ipynb)
## [3. 画像ファイルの増幅](https://github.com/mori5602/sakamotsu/blob/master/03_duplicatePictureFile.ipynb)
## [4. 学習スタート](https://github.com/mori5602/sakamotsu/blob/master/04_startTrainingSakamotsu_lecture.ipynb)
## [5. AIの学習結果を試してみる](https://github.com/mori5602/sakamotsu/blob/master/05_startSakamotsu_lecture.ipynb)

## 6. 学習を長時間繰り返す  

人工知能用のサーバにsshで接続後、screenを使って学習を繰り返し続けて放置します。
```
screen
python 04_startTrainingSakamotsu.py
```

screenのセッションから抜ける(デタッチ)場合は、以下を実行します。
```
Ctrl-a d # Ctrlを押しながらa、その後d(Ctrlは離して)
```

再度、様子を見たい(screenのセッションにアタッチしたい)ときは、以下の通り
```
screen -ls #セッションを確認する。
screen -r <セッション番号>
```

screenを終了するときは、セッションに以下の通り
```
Ctrl-a k # Ctrlを押しながらa、 その後k(Ctrlは離して)
```
