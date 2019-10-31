---
layout: post
title: Bist parser 论文和代码阅读
categories: parsing
description: bidirectional LSTM parser 依存句法解析模型， 原理和相关论文， 代码解释， 实际测试表现。
keywords: treebank, parsing
---  
 
## 引

自从 (Chen and Manning, 2014) 年的神经网络架构的 Dependency Parser 出来之后， Dependency Parsing 使用神经网络架构这件事算是开山成功。之后为了修正模型的贪心解码带来的错误传播的问题， 许多文章在解码算法上下了许多工夫。主要的改进在三个方面：

1. 贪心解码， 加上全局训练。这种方法的思路是: 贪心解码比较盲目， 应该在整个句子上训练最优的转移动作序列。主要的方法有两个， 一个是非概率的模型， 是一个在训练时使用 Structured Perceptron & Beam search & early update， 在解码时候使用 Beam Search 的方法 (Weiss et al, 2015), 它使用的 Loss 是 Strctured Perceptron 的 Loss, 当正确的转移动作序列落出 Beam 的时候更新参数; 另一个是概率模型， 是一种用 Beam 来估计配分函数的方法， 首先是得到每个转移序列的得分， 然后用它们来 normalize 出一个概率来， 然后优化正确转移序列的概率， 有时候正阳的方法被成为 Globally normalized, 或者 Maximum likelihood, 或者 probabilistic, 总之关键在于它是一个概率模型； 这方面的两篇论文有 (Zhou et al, 2015; Andor et al, 2016). 全局训练这方面的论文主要都和 (Collins, 2004) 的一篇非常高瞻远瞩的研究联系在一起。

2. Dynamic oracle 以及 Explorative Training. 这个方法的思路和上面的思路稍微有差别， 不是增加搜索的宽度， 而是把解决问题的重点放在训练阶段， 让 parser 可能面对中间已经出了错的 states, 而让 parser 在即使已经出了一些错的情况下仍然能够作出当前情况下最优的选择， 而不至于让错误传播得太过过分。在解码的情况下， 仍然维持使用简单的 greedy 解码。它的想法也很自然， 不能总是保证不出错， 出了错不扩大也已经是比较好的了。但是从效果来看， Beam Search 仍然是比较好的， 因为 Beam 比较大， 那么掉出去的可能性更小， 只要它始终没有掉出 Beam, 我们总能希望转移动作之间的全局制约 （亦即， 理想中的情况是: 每个转移动作都 conditioned on previous transitions） 能够把最优的序列找到。具体的论文可以参考 (Goldberg & Nivre, 2012, 2013) 和 (Ballesteros et al, 2016). Dynamic Oracle 的研究也是非常丰富的， 而且大多数都和 Goldberg 联系在一起。

3. Easy-First 解码方式。这种方式的思路是既然贪心解码容易错误传播，那么可以先易后难。把最优把握的那部分先解决就好， 接下来我们进行推断的时候就会有更多的根据， 可能就会更加容易一些。所以 Easy-First 解码方法放弃了从左到右这样的解码方式， 而是选择了一种哪里简单就从哪里开始的方式。这里有一个值得注意的点是， 既然要决定哪里才是最简单的决策， 那么算法就已经接近图算法的复杂度， 因为可能的决策点的个数和句子的长度相关， 而且要决策的次数也和句子长度相关。我们可以联想一种相邻二支 merge 的 parse 方法 （意想不到的  linguistic 呢！）， 那么只要 merge (n-1) 词就可以完成 parse, 但是每次都要在 O(n) 个可能的选择中选择最容易者。一般来说， 复杂度就是 O(n2). 但是通过一些 cache 的方法以及限制相邻合并， 我们可以使得复杂度在 O(nlogn). 不过仔细地看我们就会发现， Easy-First 是一种解码行为， 在训练的时候呢？ 我们需要告诉它什么是最容易的吗？ 如果是， 那岂不是又弄出来一个 static oracle 了？ 然而， 实际上不会这么操作， 本来就有多个可能的位置都是合适的， 并不能准确地说出哪里是最合适的。并且这个框架中也仍然可以使用 Explorative Training 的方式来使得 parser 更加准确。这方面的进展， 再一次， 很多都和 Goldberg 联系在一起。

以上都可以说是对 incremental parsing 的改进， 在解码算法和参数训练两个方面都有很多进展。但是 incremental parsing, 尤其是 transition-based parsing 本身的特点是重视特征表现 (Feature Representation)。尤其是 incremental feature represaentation， 依赖逐步变化的状态来提取特征， 这样的习惯使得我们的重视点常常在于对 parse state 的更好的编码。尤其是 explicitly capture partial tree structure, 或者

1. 通过 position templates exploiting partial tree structure， 比如 (Chen & Manning, 2014);
2. 或者是通过 information through structures, 比如 (Dyer et al, 2015).

但是本文所讲的这篇文章却让我们思考， 非显式的树结构所能够提供的信息也是非常丰富的， 对序列性上下文足够好的的建模也可以隐式地表达上下文的信息， 而且这样的建模在网络结构上有它的优势， 那就是可以实现简单的并行。何出此言呢？ 因为本文中讲到的 Bidirectional LSTM Feature Representation 是一个在 incremental parsing 开始之前的一个编码， 把句子中的每个词从 lookup table 中查找到的词向量通过一个双向 LSTM, 在每一个 time step, 又再次得到了一个向量。在这里我们可以反用对数螺线的一句话， “纵使改变,依然故我”， 当然这是说对数螺线的旋转不变性。而这里我们就可以说， 经过了一次变换 (用 dynet API 中对 RNN 的这个动作的隐喻， 则是传导 (transduce)), 它还在这个位置， 但是它已经不是从 lookup table 中查出来的那个 context-agnostic 的表示， 而是融汇了上下文信息的表示， 是一种 word in context 的表示。从单独的词表示序列， 到置于上下文中的表示的这个过程， 我倾向于称之为 contextualization. 而且主要是用来指对线性上下文的建模 (尽管其中可能包含了隐式的结构信息。) 这样的序列建模的关联， 更明确地出现在语言模型 (Language Modeling) 上， 而这也成为了 seq2seq translation 的一个标准组件。所以 parsing 从中吸取灵感也是很自然的事情了。

那么接下来我们来看看， 为什么 bi-LSTM 在上下文建模上这么有效呢？ 首先 LSTM 带有输入， 遗忘和输出门， 这些机制允许它可以有选择地传递信息， 而且这些门的控制和输入和以往状态有关， 而且是可以训练的， 所以比普通的 RNN 表现力更强， 在应对梯度消失的问题上更加游刃有余。其次， 双向的结构使得每个词的 contextualized representation 都和整个句子有关， 所以表现力比较强。信息的传递可以装满整个 grid, 所以从这里长出一棵树当然也没有什么奇怪了。

## 模型结构和 Trick

这两个部分一起讲吧， 这样子比较不容易顾此失彼。


### Embedding
Embedding， 包含词向量（也可以使用预训练的词向量）， 词性的嵌入。
1. 实际实验中使用了和 (Dyer et al, 2015) 一样的 external embedding. 
2. POS 使用了和 (Dyer et al, 2015) 中一样的外部 tagger 预测的词性标记。
3. word\_dropout, 或者称为 UNK\_replace.(Iyyer et al, 2015) 中的做法， 在训练的时候， 所有的词都有一定的概率被替换为 UNK， 概率和词频的关系是 \(p_{unk}(w) = \frac{\alpha}{f_w + \alpha}\) .而且如果词被 drop 成了 unk, 那么外部的 embedding 也被以 0.5 的概率被 drop.
4. 和 Dyer 一样， 随机初始化的词向量和预训练的词向量是拼接在一起的， 而且外部词向量也参与更新。
5. 数值设定， 词向量维度 100， POS 向量维度 25, unk replace 参数 alpha 0.25.

### biLSTM Encoder
1. 使用双层 bi LSTM; 隐藏状态维度 125.

### 基于转移的算法的细节

![biLSTM transition parser](/assets/images/kiperwasser_bilstm_transition.png)

1. Arc-Hybrid 转移系统。
2. 状态表示。\(\phi(c) = v_{s_2} v_{s_1} v_{s_0} v_{b_0}\) , \(v_i = BILSTM(x_{1:n}, i)\), 意思就是说特征表现就直接使用了 4 个位置模板， 提取这四个位置的 bilstm 输出， 拼接在一起就 OK 了。
3. 评分函数 \(score_{\theta}(x, t) = MLP_{\theta}(x)[t]\). 
4. 解码算法， 贪心解码， 每次都选择得分高的那个 transition.
5. 训练时候的 Dynamic oracle 和 Explorative Training.

    一般来说， 在多分类任务中， 我们会定义 hinge loss, \(max(0, m - score_{y_g} + max_{y_i \neq y_g} score(y_i))\), 意思是只要正确分类的得分不比错误分类中得分最高的那一类高出 m， 那么就要进行参数更新。这是一种经典的 one-one 形式的 hinge loss, 当然也还有 one to rest 的形式， 是正确分类和每一个错误分类都算一下 hinge loss 然后求和。但是我们一般不用那一种啦。
    
    然后在 transition-based parsing 问题中，实际的场景就是在一个 parse state 下选择最合适的转移的问题。这同样也是一个多分类问题， 但 dynamic oracle 告诉我们的是， 正确的句法树可以由多个转移动作序列来得到， 所以在具体的某个状态下的时候，我们事实上是可以选择多个可能的转移动作的。这就有点类似 multi-label 的问题了。
    
    这篇文章中使用了一种稍微特别的 one to one 形式的 hinge loss. gold label 不再是一个确定的值， 而是一个 G 集合。 loss 的定义如下
    
    \[ max(0, 1 - max_{t_o \in G}MLP(\phi(c))[t_o] + max_{t_{p} \in A\\G}MLP(\phi(c))[t_p]) \]
    
    意思就是使得 Gold 转移动作中得分最高的那个的分数要比 non-Gold 转移动作中得分自高的那个的分数高 1. G 的定义是动态的， 意思是说即使已经出了错， 这个状态下不可能达到完美的句法树了， 也仍然要定义这种情况下的 G. 所以 oracle 是在 parsing 的过程中动态计算的， 而不是根据转移动作一开始就推导好的。所以这个如果做好的， 那么我们就可以不必在构建句子的时候人为地产生一个 transitions 字段。但是这也要算法能够高效地计算出合适的 G, 最好能够在 O(1) 的时间内就可以计算出来。 Arc-Eager 和 Arc-Hybrid 是符合这样的要求的， 而 Arc-Standrad 不符合。
    
    定义好了之后， 训练的时候， parser 是直接按照得分最高的转移动作来走的， 而不是用 teacher forcing 来逐步纠正。事实上， 也可以使用概率化的方法来采样。
    
6.  激进探索。根据论文所说， 即使是在训练过程中也贪心， 模型也会很快就达到比较高的准确率， 从而使得探索比较少， 所以实验中还使用了激进探索的方式。当 non-gold 的转移动作分数低于 gold 转移动作的分数但是差距不大于 margin 常数的时候， 以一定的概率 \(p_{agg}=0.1\) 采用 non-gold 的转移动作。但是这里依然留下了一些模糊的地方， 差距是怎么计算的， gold 的最低分减去 non-gold 的最高分呢还是怎样， non-gold 的选择也很多啊， 是随机选一个呢还是怎样， 这些细节要到代码里面去才能看到了。

### 基于图的算法的细节

![biLSTM graph parser](/assets/images/kiperwasser_bilstm_graph.png)

基于图的算法和基于转移的算法一样都使用双向 LSTM 作为 contextualization. 因为使用的简单的一阶图算法， 也就是 Arc-factored 算法， 或者称为 Edge-factored 算法。是因为对一个图的评分被分解为对子图的评分的和。而一阶算法中， 每个子图都只包含一条边， 所以就叫作 arc-factored 算法。对树的评分的定义如下：

\[ parse(s) = argmax_{y \in \mathcal{Y}(s)} \sum_{(h,m) \in \mathcal{y}} score(\phi(s, h, m))\]

在这里， 一些的图算法都追踪到了 (MacDonald et al, 2005) 那篇论文。

1. 边的表示函数。使用的是 head 和 dependent 的 bilstm 输出的拼接。 \(\phi(s, h, m) = BIRNN(x_{1:n}, h) \circ BIRNN(x_{1:n}, m)\). 整个数的评分函数就是 

    \[ score_{global}(s, y) = \sum_{(h,m) \in y} score(\phi(s, h, m)) = \sum_{(h,m) \in y}MLP(v_h \circ v_m)\]
    其中 \(v_i = BIRNN(x_{1:n}, i)\)
    
2. 事实上这篇文章的图算法也有对 lstm 出来之后的词用两个 MLP 分别提取特征得到用于头和用于依存的两个不同的向量， 这也不是 deep biaffine 的独创， 所以 deep biaffine 的特别之出就是那个神奇的 biaffine 和大量的 dropout。
    
3. 目标函数是一个 hinge loss， 事实上是一个标准的多分类的 hinge loss, 但是可能使用结构化的 hinge loss 会更好。

    \[ max(0, 1 - \sum_{(h,m) \in y}MLP(v_h \circ v_m)) + max_{y^{'} \neq y} \sum_{(h, m) \in y^{'}} MLP(v_h \circ v_m)\]
    
    这里有一个问题就是无论句子多长， margin 都是一样的， 这不太合理， 换成正确的树和得分高的错误图之间的 hamming loss 似乎更为合理， 那样子就称为 structured hinge loss 了。
    
3. 对于 Labeled Parsing， 图算法一般都是结构和标签的预测分开的。用于确定依存标签的函数是

    \[ label(h,m) = argmax_{l \in labels} MLP_{lbl} (v_h \circ v_m)[l]\]
    
    这一步分是在 gold - tree 上训练的， loss 应该是使用 neglogprob.
    
4. 损失增益。这个是 (Taskar et al, 2015) 的方法， hinge loss 计算的时候不仅仅和得分最高的那个错误树对比， 也和错很多的那些树对比。我觉得这样子下去还不如就直接用 naive argmax 了， 也就是独立的头节点选取。增益之后的 loss 函数如下：

    \[ max(0, 1 - score(x, y) + max_{y^{'} \neq y} \sum_{part \in y^{'}}(score_{locqal}(x, part) + \mathbb{1}_{part \notin y})) \\]
    
    经过这么一番改动，就已经类似 structured hinge loss 了， 因为 margin 可以视为给错误类标签的补偿， 而在结构预测例子里， 错的部分越多， 那么补偿的也应该越多， 所以标准的 structured hinge loss 用 Hamming Loss 作为补偿。

## 实验结果

在 PTB 3 和 CTB 5 上的实验结果如图。

![biLSTM bilstm results](/assets/images/kiperwasser_bilstm_results.png)

内部对比结论是：

1. 没有使用外部词嵌入的情况下， 英文是 transition 加上额外位置模板的分数比较高， 而中文则是一阶图算法比较高。
2. 带上外部词嵌入的条件下， 一阶图算法反而衰退， 而转移算法则能够更好地得到提升。果然应了那句老话， 转移算法重特征表示， 图算法重解码。
3. 带上了外部词向量和额外位置模板之后， 这个算法和 Weiss 的 beam search 结果相当， 超过了 Dyer 15, 这是 PTB 上的结果。
4. 在 CTB 上， 带有外部词向量和额外位置模板的转移 parser， 几乎追平 Ballesteros 的结果 (也就是 Dyer 的模型带上 dynamic oracle 的结果).

参数设置如图：

![biLSTM bilstm hp](/assets/images/kiperwasser_bilstm_hp.png)

## 去除分析

![biLSTM bilstm ablation](/assets/images/kiperwasser_bilstm_ablation.png)

### 图算法
1. 对于图算法， loss augmentation 也就是 structured hinge loss 很重要。
2. arc labeler 对于 UAS 也有提升。
3. POS 对于汉语太重要了。

### 转移算法
1. dynamic oracle 是很有用的。


