# VAEGAN for airfoil generation

深層生成モデルVAEGANを用いた翼形状生成


## Features

2021年度東京大学工学部卒業研究「Conditional-VAEGANの潜在空間表現力を活かした翼形状生成」で用いた実験ソースコードです。<br>
詳細はこちらの[卒業研究発表用スライド](https://github.com/YukiTomori-starrr/VAEGAN_airfoil/files/8137821/_.ver.pdf)をご覧ください。

## Requirement

本研究では、Deep Learning Box Alpha (GDEP製)・GPU: NVIDIA RTX A6000 (48GB memory)を用いています。<br> 
Anaconda等で仮想環境を構築して実行してください。なお、仮想環境のrequirementは以下です。

```
# Name                    Version                   Build
_libgcc_mutex             0.1                        main  
_openmp_mutex             4.5                       1_gnu  
blas                      1.0                         mkl  
bottleneck                1.3.2            py39hdd57654_1  
ca-certificates           2021.10.26           h06a4308_2  
certifi                   2021.10.8        py39h06a4308_0  
cycler                    0.10.0           py39h06a4308_0  
dbus                      1.13.18              hb2f20db_0  
deprecated                1.2.12             pyhd3eb1b0_0  
dotmap                    1.3.23                   pypi_0
expat                     2.4.1                h2531618_2  
fontconfig                2.13.1               h6c09931_0  
freetype                  2.10.4               h5ab3b9f_0  
glib                      2.68.2               h36276a3_0  
gst-plugins-base          1.14.0               h8213a91_2  
gstreamer                 1.14.0               h28cd5cc_2  
icu                       58.2                 he6710b0_3  
intel-openmp              2021.2.0           h06a4308_610  
joblib                    1.1.0                    pypi_0
jpeg                      9b                   h024ee3a_2  
kiwisolver                1.3.1            py39h2531618_0  
lcms2                     2.12                 h3be6417_0  
ld_impl_linux-64          2.35.1               h7274673_9  
libffi                    3.3                  he6710b0_2  
libgcc-ng                 9.3.0               h5101ec6_17  
libgfortran-ng            7.5.0               ha8ba4b0_17  
libgfortran4              7.5.0               ha8ba4b0_17  
libgomp                   9.3.0               h5101ec6_17  
libpng                    1.6.37               hbc83047_0  
libstdcxx-ng              9.3.0               hd4cf53a_17  
libtiff                   4.2.0                h85742a9_0  
libuuid                   1.0.3                h1bed415_2  
libwebp-base              1.2.0                h27cfd23_0  
libxcb                    1.14                 h7b6447c_0  
libxml2                   2.9.12               h03d6c58_0  
lz4-c                     1.9.3                h2531618_0  
matplotlib                3.3.4            py39h06a4308_0  
matplotlib-base           3.3.4            py39h62a2d02_0  
mkl                       2021.2.0           h06a4308_296  
mkl-service               2.3.0            py39h27cfd23_1  
mkl_fft                   1.3.0            py39h42c9631_2  
mkl_random                1.2.1            py39ha9443f7_2  
ncurses                   6.2                  he6710b0_1  
numexpr                   2.7.3            py39h22e1b3c_1  
numpy                     1.21.0                   pypi_0
numpy-base                1.20.2           py39hfae3a4d_0  
olefile                   0.46                       py_0  
openssl                   1.1.1l               h7f8727e_0  
pandas                    1.2.5            py39h295c915_0  
pcre                      8.45                 h295c915_0  
pillow                    8.2.0            py39he98fc37_0  
pip                       21.1.2           py39h06a4308_0  
pyparsing                 2.4.7              pyhd3eb1b0_0  
pyqt                      5.9.2            py39h2531618_6  
python                    3.9.5                h12debd9_4  
python-dateutil           2.8.1              pyhd3eb1b0_0  
pytz                      2021.1             pyhd3eb1b0_0  
qt                        5.9.7                h5867ecd_1  
readline                  8.1                  h27cfd23_0  
scikit-learn              1.0.1                    pypi_0 
scipy                     1.6.2            py39had2a1c9_1  
seaborn                   0.11.2             pyhd3eb1b0_0  
setuptools                52.0.0           py39h06a4308_0  
sip                       4.19.13          py39h2531618_0  
six                       1.16.0             pyhd3eb1b0_0  
sklearn                   0.0                      pypi_0   
sqlite                    3.36.0               hc218d9a_0  
threadpoolctl             3.0.0                    pypi_0  
tk                        8.6.10               hbc83047_0  
torch                     1.10.0.dev20210629+cu111          pypi_0 
torchaudio                0.10.0.dev20210629          pypi_0  
torchvision               0.11.0.dev20210629+cu111          pypi_0 
tornado                   6.1              py39h27cfd23_0  
typing-extensions         3.10.0.0                 pypi_0 
tzdata                    2021a                h52ac0ba_0  
wheel                     0.36.2             pyhd3eb1b0_0  
wrapt                     1.12.1           py39he8ac12f_1  
xfoil                     0.0.16                   pypi_0  
xz                        5.2.5                h7b6447c_0  
zlib                      1.2.11               h7b6447c_3  
zstd                      1.4.9                haebb681_0 
```
## Installation

深層学習フレームワークの[pytorch](https://pytorch.org/)と形状解析ソフトXFoilのPython用ライブラリ[xfoil-python](https://github.com/KikeM/xfoil-python)のインストールが必要です。

### Pytorch

`train.py`を実行してモデルを学習させる際に、深層学習フレームワーク[pytorch](https://pytorch.org/)を用います。

### xfoil-python

生成した形状の揚力係数を再計算する際（evalファイルなどを実行し`calc_cl.py`を動かすとき）に、形状解析ソフト`XFoil`の[python用ライブラリ](https://github.com/KikeM/xfoil-python)を用います。
しかし、こちらのインストールにはかなり苦戦したので[こちら（miyamotononno/generate_airfoilより引用）](https://github.com/miyamotononno/generate_airfoil/issues/14)を参考にしてください。

## Usage

代表的なコマンドを以下に記します。
- 通常のConditional VAEGAN
```
#訓練
python3 -m vaegan.train

#評価
python3 -m vaeagan.eval
```
- cWVAEGAN-gp(encoderにラベル情報を与えるver：提案手法)
```
#訓練
python3 -m wvaegan_gp.train_vector

#評価
python3 -m wvaegan_gp.eval_vector

#潜在空間の評価
python3 -m wvaegan_gp.latent_eval_vector
```
- eval_vectorの使い方
    - `calc_values(self)`：converge, failure, smoothness, mse, muを算出する関数
    - `plot_multicoords(self, clc)`：生成翼型を重ねて表示させる関数、形状の多様性の評価時に用いる
        -  `clc`:生成した重ねたい翼型

- latent_eval_vectorの使い方
    - `visualize_latentspace(dim=3, latent_vector=mu)`：潜在空間の可視化を行う関数
         - `dim`：t-sneで可視化する潜在空間の次元数、`dim=2`, `dim=3`など
         - `latent_vector`：可視化する潜在ベクトル、`latent_vector=mu`,`latent_vector=log_variances`など


- cWVAEGAN-gp(encoderにラベル情報を与えないver)
```
#訓練
python3 -m wvaegan_gp.train_nonvector

#評価
python3 -m wvaegan_gp.eval_nonvector

#潜在空間の評価
python3 -m wvaegan_gp.latent_eval_nonvector
```

## Files

### dataset
翼型訓練データ
* coords：翼型の点列データセット
* perfs：翼型に対応する揚力係数データセット

### vaegan
通常のConditional VAEGANのモデル定義・訓練・評価

### wvaegan_gp
cWVAEGAN-gpのモデル定義・訓練・評価

## Reference
* VAEGANの元論文：[Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/abs/1512.09300)
* 参考コード1：【github】[miyamotononno/generate_airfoil](https://github.com/miyamotononno/generate_airfoil)
* 参考コード2：【github】[lucabergamini/VAEGAN-PYTORCH](https://github.com/lucabergamini/VAEGAN-PYTORCH)

## Author
* 友利優希（Yuki Tomori）
* 東京大学工学部システム創成学科B4　鈴木・米倉研究室（2022年時点）
