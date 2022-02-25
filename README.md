# VAEGAN for airfoil generation

深層生成モデルVAEGANを用いた翼形状生成


## Features

2021年度東京大学工学部卒業研究「Conditional-VAEGANの潜在空間表現力を活かした翼形状生成」で用いた実験ソースコードです。<br>
詳細はこちらの[卒業論文]()と[卒業研究発表用スライド]()をご覧ください。

## Requirement

本研究では、Deep Learning Box Alpha (GDEP製)・GPU: NVIDIA RTX A6000 (48GB memory)を用いています。<br> 
Anaconda等で仮想環境を構築して実行してください。なお、仮想環境のrequirementは以下です。

* python 3.7.
* pytorch
* xfoil
* numpy
* pandas
* 

## Installation

深層学習フレームワークの[pytorch](https://pytorch.org/)と形状解析ソフトXFoilのPython用ライブラリ[xfoil-python](https://github.com/KikeM/xfoil-python)のインストールが必要です。

### Pytorch

train.pyを実行してモデルを学習させる際に、深層学習フレームワークpytorchを用います。

### xfoil-python

生成した形状の揚力係数を再計算する際（evalファイルなどを実行しcalc_cl.pyを動かすとき）に、形状解析ソフトXFoilを用います。
しかし、こちらのインストールにはかなり苦戦したので[こちら（〜より引用）](https://github.com/miyamotononno/generate_airfoil/issues/14)を参考にしてください。

```
aaa
pip install mecab-python3
```

# Usage

代表的なコマンドを以下に記します。
- 通常のConditional VAEGAN
```
#訓練
python3 -m vaegan.train

#評価
python3 -m vaeagan.eval
```
- cWVAEGAN-gp(encoderにラベル情報を与えるver)
```
#訓練
python3 -m wvaegan_gp.train_vector

#評価
python3 -m wvaegan_gp.eval_vector

#潜在空間の評価
python3 -m wvaegan_gp.latent_eval_vector
```
- cWVAEGAN-gp(encoderにラベル情報を与えないver)
```
#訓練
python3 -m wvaegan_gp.train_nonvector

#評価
python3 -m wvaegan_gp.eval_nonvector

#潜在空間の評価
python3 -m wvaegan_gp.latent_eval_nonvector
```

# Files

### dataset
翼型訓練データ
* coords：翼型の点列データセット
* perfs：翼型に対応する揚力係数データセット

### vaegan
通常のConditional VAEGANのモデル定義・訓練・評価

### wvaegan_gp
cWVAEGAN-gpのモデル定義・訓練・評価

# Author
* 友利優希（Yuki Tomori）
* 東京大学工学部システム創成学科B4（2022年時点）
