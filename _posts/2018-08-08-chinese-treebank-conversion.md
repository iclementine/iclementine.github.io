---
layout: post
title: 中文树库转换
categories: treebank
description: 关于中文宾州树库 LDC CTB, 转换成为依存树库的各种方式。 SD, Penn2Malt, 以及规则文件.
keywords: treebank, parsing
--- 


首先， 进行树库转换主要是为了利用已有的标注成果， 进行依存句法解析研究的时候得到的格式是句子中词和词的依存关系以及依存关系类型。常见的格式有 Malt-Tab 格式， Conll-X 格式， Cobll-U 格式。

## 关于树库转换

Malt-Tab 的详细说明可以参见 [Malt Converter](http://stp.lingfil.uu.se/~nivre/research/MaltXML.html). 格式是 4 列表格。
`form (required) < postag (required) < head (optional) < deprel (optional)`.

格式示例：

```
Den	pn.utr.sin.def.sub/obj	2	SUB
blir	vb.prs.akt.kop	0	ROOT
gemensam	jj.pos.utr.sin.ind.nom	2	PRD
för	pp	2	ADV
alla	dt.utr/neu.plu.ind/def	6	DET
inkomsttagare	nn.utr.plu.ind.nom	4	PR
oavsett	jj.pos.neu.sin.ind.nom	2	ADV
civilstånd	nn.neu.sin.ind.nom	7	PR
.	mad	2	IP
```

Conll-X 格式则可以参考 [CoNLL-X shared task on Multilingual Dependency Parsing](http://anthology.aclweb.org/W/W06/W06-2920.pdf). 是一个 10 列表格

`id < form < lemma < cpostag < postag < feats < head < deprel < phead(predicted) < pdeprel(predicted)`

其中最后的两列是预测的结果， 所以实际上是 8 列表格。

Conllu-U 格式可以参考 [Conllu-U Format](http://universaldependencies.org/format.html).

1. ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes.
2. FORM: Word form or punctuation symbol.
3. LEMMA: Lemma or stem of word form.
4. UPOS: Universal part-of-speech tag.
5. XPOS: Language-specific part-of-speech tag; underscore if not available.
6. FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
7.HEAD: Head of the current word, which is either a value of ID or zero (0).
8.DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
9. DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
10. MISC: Any other annotation.

## 关于树库转换程序

### stanford dependency conversion
[Stanford Dependencies](https://nlp.stanford.edu/software/stanford-dependencies.shtml) 说明了可以使用它的代码来进行树库转换。它是 [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.html#Download) 的一部分。

那么我们主要看其中关于中文树库转换的部分

>SD for Chinese
Stanford dependencies are also available for Chinese. The Chinese dependencies have been developed by Huihsin Tseng and Pi-Chuan Chang. A brief description of the Chinese grammatical relations can be found in this paper.

>If you have a version of the LDC Chinese Treebank (or some other Chinese constituency treebank in Penn Treebank s-expression format) in the file or directory treebank, you can use our code to convert it to a file of basic Chinse Stanford Dependencies in CoNLL-X format with this command:

```{bash}
java -mx1g edu.stanford.nlp.trees.international.pennchinese.ChineseGrammaticalStructure -basic -keepPunct -conllx -treeFile treebank > treebank.csd
```

通过下载文件之后， 查看各个 `.jar` 文件内部的结构， 发现 `edu.stanford.nlp.trees.international.pennchinese.ChineseGrammaticalStructure` 是在 `stanford-parser.jar` 里面。当然， 我还以为这个是在 `stanford-parser-3.9.1-models.jar` 里面， 结果找不到， 或者是在 `stanford-chinese-corenlp-2018-02-27-models.jar` 里面， 原来并非如此。(事实上我还为此下载了整个 corenlp).

然后通过临时把这个 `stanford-parser.jar` 加入 `$CLASSPATH` 里面的方式， 我们就可以运行里面的 `class` 字节码了。

方法1

```{bash}
java -cp stanford-parser.jar -mx1g edu.stanford.nlp.trees.international.pennchinese.ChineseGrammaticalStructure -basic -keepPunct -conllx -treeFile treebank > treebank.csd
```

方法2

```{bash}
export CLASSPATH=./stanford-parser.jar
java -mx1g edu.stanford.nlp.trees.international.pennchinese.ChineseGrammaticalStructure -basic -keepPunct -conllx -treeFile treebank > treebank.csd
```

这样都可以做到， 那么就得到了 SD 标注体系的依存树库， 使用的格式是 Conll-X 的十列表格式。 注意的是 输入的 treeban 需要为每个句子套一层额外的括号， 使得每个句子都是一个 unary. 比如说像下面这样子。

```
( (S(NP ...)(VP ...)) )
```

这个方法转换得到的树库是 OK 的了！

### Penn2Malt 转换程序

[Penn2Malt](http://stp.lingfil.uu.se/~nivre/research/Penn2Malt.html) 是 Joakim Nivre 的成果， 应该是当年为了开发 MaltParser 作的工作。其中关于 deprel 的控制项有三个选项。

>
<deprel> is a flag that determines which dependency labels will be used; there are currently three options:
1 = Penn labels: phrase label + head label + dependent label (à la Collins)
2 = Penn labels: dependent label only
3 = Malt: hard-coded mapping to dependency labels (SBJ, OBJ, PRD, NMOD, VMOD, etc.)
When Penn labels are used, only phrase labels are used, except that -SBJ and -PRD are retained on NPs, and -OBJ is added to NPs under VP that lack an adverbial function tag.

一般都是选择 3 的啦。因为这样使用的是比较简洁的 deprel 体系， 而以上的两种都是靠短语结构树库中的信息拼接出来的， 里面并没有简洁的 deprel.



## 关于转换规则

[Chen and Manning, 2014](https://cs.stanford.edu/~danqi/papers/emnlp2014.pdf) 中使用的树库转换方式就是使用 Penn2Malt， 使用了 [Zhang and Clark, 2008](http://www.aclweb.org/anthology/D/D08/D08-1059.pdf) 中使用的头节点抽取规则， 这个规则也可以在张岳的 [个人页面](http://www.cs.ox.ac.uk/people/yue.zhang/ctbheadfinding.html) 找到. 一共 22 条， 在论文中还有说明包含一条 default 用来处理没有匹配到的情况， 那样子就 23 条。

>
ADJP	: r ADJP JJ AD; r 
ADVP	: r ADVP AD CS JJ NP PP P VA VV; r 
CLP	: r CLP M NN NP; r 
CP	: r CP IP VP; r 
DNP	: r DEG DNP DEC QP; r 
DP	: r M; l DP DT OD; l 
DVP	: r DEV AD VP; r 
IP	: r VP IP NP; r 
LCP	: r LCP LC; r 
LST	: r CD NP QP; r 
NP	: r NP NN IP NR NT; r 
NN	: r NP NN IP NR NT; r 
PP	: l P PP; l 
PRN	: l PU; l 
QP	: r QP CLP CD; r 
UCP	: l IP NP VP; l 
VCD	: l VV VA VE; l 
VP	: l VE VC VV VNV VPT VRD VSB VCD VP; l 
VPT	: l VA VV; l 
VRD	: l VV VA; l 
VSB	: r VV VE; r 
FRAG : r VV NR NN NT; r 

而在 (Zhang and Clark, 2008) 中, 又有提及到这个规则文件主要来源。

> Most of the head-finding rules are from Sun and Jurafsky (2004), while we added rules to handle NN and FRAG, and a default rule to use the rightmost node as the head for the constituent that are not listed.

那么 [Sun and Jurafsky, 2004](https://web.stanford.edu/~jurafsky/Sun-Jurafsky-HLT-NAACL04.pdf), 是一片关于汉语的浅层语义分析的文章， 其中的规则是是样子的呢？

>
Parent  Direction   Priority List
ADJP    Right   ADJP JJ AD
ADVP    Right   ADVP AD CS JJ NP PP P VA VV
CLP Right   CLP M NN NP
CP  Right   CP IP VP
DNP Right   DEG DNP DEC QP
DP  Left    M(r) DP DT OD
DVP Right   DEV AD VP
IP  Right   VP IP NP
LCP Right   LCP LC
LST Right   CD NP QP
NP  Right   NP NN IP NR NT
PP  Left    P PP
PRN Left    PU
QP  Right   QP CLP CD
UCP Left    IP NP VP
VCD Left    VV VA VE
VP  Left    VE VC VV VNV VPT VRD VSB VCD VP
VPT Left    VA VV
VRD Left    VVl VA
VSB Right   VV VE 

大约的意义也是比较清楚的， 前面是一种 Constituent 的类型， 如果 Direction 是 Left, 那么就从左往右扫描， 从 Priority List 最高优先级的开始选， 而且这个规则是递归的， 比如 `ADJP` 最有限的还是 `ADJP` 那么就跳入下一层进行选取， 以此类推。一共 20 条规则。

但是我在使用过程中遇到的问题是， 尽管头节点的选取没有问题， 但是关于 deprel 的判断的结果却比较差， 因为这毕竟是它根据 head, dependent 来推断出来的。 常常会出现一些没有具体依存类型的 `DEP` 标记， 这样的标记太过粗糙，而且也会带来类不平均的问题的吧。

## 仍然存在的问题

无论是使用 stanford 转换， 还是 Penn2Malt 带上 (Zhang and Clark, 2008) 的规则， 都会遇到 `DEP` 这种泛用类型的事情， 这种标准对于用提升 LAS 来提升 UAS 这个思路应该不是一个好的因素。

Work in Progress!

