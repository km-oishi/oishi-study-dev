# Chapter 4 確率的生成モデルとガウス過程

「観測 $Y$ は確率分布 $p(Y)$ からのサンプリング $Y \sim p(Y)$ により得られた」とする仮説を、観測 $Y$ の確率的生成モデル（probabilistic generative model）と呼ぶ。本章ではこのモデリングの考え方、定式化、計算方法を扱い、基礎固めを行う。

ガウス過程回帰の基本モデル $y = f(\mathbf{x}) + \epsilon$ では、入力を定数 $\mathbf{x}$ と固定した上で、関数 $f(\cdot)$、観測値 $y$、観測ノイズ $\epsilon$ を確率変数とみなす。ここには直感と逆行する2つの大きな**発想の転換**がある。

* **尤度関数の導入**：「目の前にある観測値 $y$ は確率変数である」という転換（最尤推定法の基礎）。
* **事前確率・事後確率の導入**：「知りたい対象（隠れ変数 $\mathbf{f}$ や未知パラメータ $\theta$）は確率変数である」という転換（ベイズ推定の基礎）。

これらはガウス過程法のみならず機械学習全般の根底をなす重要な考え方である。

---

## 4.1 確率変数と確率的生成モデル

### 4.1.1 確率変数 $X$ と確率分布 $p(X)$

確率変数（random variable）とは、試行ごとに値が確率的に決まる変数である。

* **離散値の場合（例 1：サイコロの出目）**
とり得る値の集合を $\{1, \dots, 6\}$ とすると、各出目の確率の和は 1 となる。

$$\sum_{i=1}^6 p(X = i) = 1$$


* **連続値の場合（例 2：ダーツの刺さった位置）**
実現値 $x \in \mathbb{R}$ の出現確率は確率密度関数 $p(x)$ で表され、特定範囲に入る確率は積分 $p(a < x < b) = \int_a^b p(x) dx$ で定義される。全区間の積分は 1 となる。

$$\int_{-\infty}^\infty p(y) dy = \int p(y) dy = 1$$



1次元ガウス分布の確率密度関数は次式で与えられる。

$$\mathcal{N}(x \vert{} \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( - \frac{1}{2\sigma^2} (x - \mu)^2 \right)$$



> **定義 4.1（確率過程）**
> 任意の $N$ 個の入力 $\mathbf{x}_1, \dots, \mathbf{x}_N \in \mathcal{X}$ に対し、$N$ 個の出力 $\mathbf{f}_N = (f(\mathbf{x}_1), \dots, f(\mathbf{x}_N))$ の同時確率 $p(\mathbf{f}_N)$ を与えられる関係 $f(\cdot)$ を**確率過程（stochastic process）**と呼ぶ。

> **定義 4.2（ガウス過程）**
> 確率過程 $f(\cdot)$ において、同時確率 $p(f_1, \dots, f_N)$ が $N$ 次元ガウス分布として得られる場合、$f(\cdot)$ を**ガウス過程**と呼ぶ。

「無限次元ベクトル」や「ヒルベルト空間」を持ち出さずとも、「任意の自然数 $N$ に対して同時確率 $p(f_1, \dots, f_N)$ が定まる」と有限の枠組みで言い換えることで、ガウス過程を実用的に扱える。

---

### 4.1.2 同時確率 $p(X, Y)$ と周辺化

複数の確率変数の組 $(X, Y) \in \mathcal{X} \times \mathcal{Y}$ の分布を**同時分布（joint distribution）** $p(X, Y)$ と呼ぶ。

> **定義 4.3（周辺化と周辺分布）**
> 同時分布 $p(X, Y)$ から不要な変数を積分（または総和）によって消去し、$p(X)$ を得る操作を**周辺化（marginalization）**、得られた分布を**周辺分布（marginal distribution）**と呼ぶ。
> 
> $$p(X) = \int p(X, Y) dY \quad \left( \text{離散値の場合: } p(X) = \sum_{Y \in \mathcal{Y}} p(X, Y) \right)$$
> 
> 

> **定義 4.4（条件付き分布）**
> $X$ を既知・所与としたときの $Y$ の確率分布を**条件付き分布** $p(Y\vert{}X)$ と呼ぶ。
> 
> $$\int p(Y\vert{}X) dY = 1 \tag{4.1}$$
> 
> 
> 
> ※ 条件側 $X$ に関する積分 $\int p(Y\vert{}X) dX$ には制約がない点に注意。

> **定義 4.5（ベイズの定理）**
> 乗法定理 $p(X, Y) = p(Y\vert{}X)p(X) = p(X\vert{}Y)p(Y)$ より導かれる：
> 
> $$p(Y\vert{}X) = \frac{p(X\vert{}Y)p(Y)}{p(X)}$$
> 
> 

#### 連鎖的生成モデルの手順（例 3：サイコロ・ダーツモデル）

サイコロの出目 $d \in \{1,\dots,6\}$ に応じて狙う位置 $\boldsymbol{\mu}(d)$ を決め、投げたダーツの刺さった位置 $x$ を観測するモデルを考える。

1. 未知の値を確率変数で表す（$x, d$）。
2. 個々の生成過程を確率分布で表す：
$$p(d) = 1/6, \quad p(x\vert{}d) = \mathcal{N}(x \vert{} \boldsymbol{\mu}(d), \sigma^2)$$



記号 $\sim$ を用いるとシンプルに表現できる：

$$\begin{cases} d \sim p(d) \\ x\vert{}d \sim \mathcal{N}(\boldsymbol{\mu}(d), \sigma^2) \end{cases} \tag{4.4}$$


3. 同時分布を表す：$p(x, d) = p(x\vert{}d)p(d)$。
4. 不要な変数を周辺化して必要な分布を求める：

$$p(x) = \sum_{d=1}^6 p(d)p(x\vert{}d) = \sum_{d=1}^6 \frac{1}{6} \mathcal{N}(x \vert{} \boldsymbol{\mu}(d), \sigma^2)$$



これは複数のガウス分布の重み付き平均であり、混合ガウス分布（mixture of Gaussian distribution）と呼ばれる。

#### ガウス過程回帰モデルへの適用（例 4）

入力点 $\mathbf{X}$ を固定したとき、ガウス過程回帰も2段階の連鎖的モデルとなる。


$$\begin{cases} \mathbf{f} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{K}) \\ \mathbf{y}\vert{}\mathbf{f} \sim \mathcal{N}(\mathbf{f}, \sigma^2 \mathbf{I}_N) \end{cases} \tag{4.5}$$


観測値 $\mathbf{y}$ の予測分布は、潜在関数値 $\mathbf{f}$ を周辺化積分して導出される：


$$p(\mathbf{y}) = \int p(\mathbf{y}\vert{}\mathbf{f}) p(\mathbf{f}) d\mathbf{f}$$

---

### 4.1.3 独立性と条件付き独立性

> **定義 4.6（確率変数の独立性）**
> 次式が成り立つとき、$X$ と $Y$ は**独立（independent）**である（$p(X) = p(X\vert{}Y)$ と等価）。
> 
> $$p(X, Y) = p(X)p(Y)$$
> 
> 

> **定義 4.7（確率変数の条件付き独立性）**
> 次式を満たすとき、$X$ と $Y$ は**条件 $Z$ のもとで条件付き独立（conditionally independent）**である。
> 
> $$p(X, Y \vert{} Z) = p(X\vert{}Z)p(Y\vert{}Z)$$
> 
> 

※ 条件付き独立であっても、無条件で独立 $p(X, Y) = p(X)p(Y)$ とは限らない点に注意。

* **多変量ガウス分布における独立性（例 6）**
共分散行列が対角行列 $\sigma^2 \mathbf{I}$ のとき、確率密度関数の指数部が分離され、$p(y_1, y_2, y_3) = p(y_1)p(y_2)p(y_3)$ となり互いに独立となる。
* **ガウス過程回帰における性質（例 7）**
$\mathbf{f}$ が与えられた下では各観測ノイズは無相関なので、観測値は条件付き独立となる：

$$p(\mathbf{y}\vert{}\mathbf{f}) = \prod_{n=1}^N p(y_n \vert{} f_n)$$



しかし、$\mathbf{f}$ を周辺化消去した $p(\mathbf{y})$ ではカーネル関数による相関が残るため、一般に $y_n$ 同士は無条件には独立ではない（$p(\mathbf{y}) \neq \prod p(y_n)$）。
* **独立同分布（例 8）**
同一分布から独立に標本を得ることを i.i.d.（independent and identically distributed）サンプリングと呼び、$d_1, d_2, d_3 \stackrel{\text{i.i.d.}}{\sim} p(d)$ と表記する。現実には完全な i.i.d. は存在せず、理解を単純化するための仮定である。

---

### 4.1.4 ガウス過程回帰モデルのグラフィカルモデル

グラフィカルモデル（graphical model）は、変数間の独立性や依存関係を可視化する記法である。

* **○（円ノード）**：確率変数。
* **□（四角ノード）**：観測データ（確定した値）。
* **文字のみ（丸なし）**：定数・所与のパラメータ（入力 $\mathbf{X}$ など）。
* **矢印（有向リンク $\rightarrow$）**：条件付き確率 $p(X\vert{}Y)$（$Y \rightarrow X$）。
* **実線（無向リンク）**：相関があり、同時確率で表される関係。
* **プレート表示（外枠パネル）**：$n = 1, \dots, N$ 回の繰り返しサンプリングを簡潔に示す表現。

#### 線形回帰モデルとの対比（例 9, 例 10）

* **線形回帰**：

$$\begin{cases}   w_m \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, \lambda^2) \\   f_n \vert{} \mathbf{x}_n = \sum_{m=1}^M w_m \phi_m(\mathbf{x}_n) \\   y_n \vert{} f_n \sim \mathcal{N}(f_n, \sigma^2)   \end{cases} \tag{4.6}$$



潜在関数値 $f_1, \dots, f_N$ は共通パラメータ $\mathbf{w}$ を介して結ばれており、$f_n$ 同士を直接結ぶリンクは存在しない。
* **ガウス過程回帰**：

$$\begin{aligned}   \mathbf{f}_N, f_* \vert{} \mathbf{X}, \mathbf{x}_* &\sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{K}) \tag{4.7} \\   y_n \vert{} f_n &\sim \mathcal{N}(f_n, \sigma^2) \tag{4.8}   \end{aligned}$$



同時確率が共分散行列 $\mathbf{K}$ で結ばれているため、**グラフィカルモデル上では $f_1, \dots, f_N, f_*$ の全ペアが無向リンクで結合される**点に特徴がある。

---

## 4.2 最尤推定とベイズ推定

### 4.2.1 確率的生成モデルと最尤推定

観測 $Y$ が分布 $p(Y)$ からサンプリングされたとする仮説を確率的生成モデル（または確率モデル）と呼ぶ。

これはあくまで現実を説明するための仮説であり、真である保証はない。麻雀のような確率的ゲームだけでなく、囲碁・将棋のような完全情報決定論的ゲームの勝敗データをモデル化する際にも用いられる。

---

### 【補足】表記法・脚注メモ

* ***1（$p(X)$ の表記）**：離散型なら確率分布、連続型なら確率密度関数を表す。実用上の混乱が少ないため、大文字 $P$ やフォント変更等の過度な区別は行わない。
* ***2（表記のバリエーション）**：$x \sim \mathcal{N}(\mu, \sigma^2)$ はサンプリング過程を強調し、$p(x\vert{}\mu) = \mathcal{N}(x \vert{} \mu, \sigma^2)$ はパラメータへの依存性を明示した書き方である。状況に応じて固定パラメータが省略される場合がある。
* ***3（ガウス過程の有限性）**：潜在関数ベクトル $\mathbf{f} = (f(\mathbf{x}_1), \dots, f(\mathbf{x}_N))$ は通常の $N$ 次元確率変数である。機械学習の実用上は常に有限個の $N$ しか扱わないため、無限次元の議論を避けられる。