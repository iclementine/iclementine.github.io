---
layout: post
title: NLTK 中树库 API 的使用
categories: nltk
description: NLTK 是一个非常丰富的文本工具箱， 我们来使用一下它的 Treebank 相关的 API。
keywords: nltk, treebank, dependency, conllu
---


首先来说说怎么加载自定义的树库数据吧。反正如果只能加载自带的树库的话， 还是不太够用， 而且因为版权的缘故， 自带的树库也只是自带了一部分。那么， 其实 NLTK 里面有封装好的 Reader， 能够读取那种 lisp 风格的语料库。

## nltk_data 文件夹

默认 nltk_data 会有几个可能的目录， 放在这里就可以找到， 使用 `nltk.download()` 的时候也会找到目前的路径然后向里面添加。那么那几个目录是：

```bash
LookupError: 
**********************************************************************
  Resource u'corpora/gutenberg' not found.  Please use the NLTK
  Downloader to obtain the resource:  >>> nltk.download()
  Searched in:
    - '/home/<username>/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/local/lib/nltk_data'
**********************************************************************
```

一般情况下， 如果 python 解释器也是系统级的， 那么也倾向于把 nltk_data 文件夹 放在 `/usr/local/share/` 这样子的话就全部都是系统级的， 并且也保证了只读而非可写入的性质。

## 设置环境变量的正确方式

而如果使用了 anaconda 并且开了很多虚拟环境， 那么其实比较方便的做法是直接就把 anaconda 安装在用户目录下， 或者至少是把新的 env 开在自己的用户目录， 那么管理起来比较方便， 不用动不动就 sudo, 而且用 `.profile` 中把 anaconda 的 bin 加入 PATH， 并且在系统的 python 之前， 就可以默认使用 anaconda 里面的 python 解释器， 并且 activate, pip 等等的工具也一系列地导入进来了。使用 source activate 和 deactivate 可以在当前的这个 shell 进程内设置环境变量以激活想要的虚拟环境， 用起来还是很方便的啦。

而且 `.profile` 只是在登录时加载一次， 不会像在 `.bashrc` 中那样， 登录中设置一次， 每次开启一个 bash 又设置一次， 如果是 `export PATH=/home/<username/anaconda3/bin:$PATH>` 这样子的话就是重复一份， 感觉不是那么好啦。只要设置一次的事情本来就是在 `.profile` 里面设置。而且这样子设置不会影响登录之前的一些对 python 的 call， 因为 sddm 或者  KDE 有一些 DBus 有 python 调用， 如果设置系统级的环境变量会导致无法正常登录系统。

另外， 有几个方法可以设置环境变量

1. /etc/profile 系统级的环境变量
2. ~/.profile 用户级的环境变量 （一般设置这个就可以了）， 登录 shell 的脚本
3. ~/.bashrc 这是其实不是正确的用法， 这个是交互 shell 的脚本

所以设置环境变量的正确方式是在 `~/.profile`.

## NLTK 中的的 Reader 和 LazyCorpusReader

果然 API 的设计和抽象都很合理， Raeder 就是一种数据 parser, 而 LazyCorpusReader 提供了懒惰加载的特征， 这个特征是和具体的 Reader 无关的， 虽然懒惰的做法可以把它实现在一起。看到好好研读 NLTK 的代码将会使得以后自己写代码变得更加优美。

```python
import nltk
from nltk.corpus import BracketParseCorpusReader, LazyCorpusReader

ctb_reader = LazyCorpusLoader('ctb', BracketParseCorpusReader, 
r'chtb_\d{4}\.\w{2}', tagset='unknownn') 
# 'ctb' 是目录名字， 默认在 nltk_data/corpora 文件夹下寻找
# 'nltk_data/corpora/ctb' 目录下默认就是各个 chtb0001.nw 之类的文件
# BracketParseCorpusReader 是一个 Reader 实例
# fileids=r'chtb_\d{4}\.\w{2}' 是一个正则表达式 pattern, 用来滤出文件夹下需要的文件
# tagset 是这个树库所使用的 tag 集

trees = ctb_reader.parsed_sents() # 就样子就可以得到句法树
tagged_sents = ctb_reader.tagged_sents() # 这样子就可以得到 tagged 的句子
tagged_words = ctb_reader.tagged_words() # 这样子就可以得到 tagged 的词， 所有句子连在一起
```

所以可见 Reader 类是一种多么丰富的存在， 还是很方便的啦。如果是我来写可能会写出多个 Reader，  然而比较好的做法应该是一个 Reader， 但是可以得到各种输出格式。于是我赶紧查了 

`nltk.corpus.reader.dependency module`

## DependencyGraph 类和 DependencyCorpusReader 类

果然厉害呀！ 我被震惊了。

DependenvyGraph


```python
class nltk.corpus.reader.DependencyCorpusReader(
root, # 这里必须是绝对路径或者相对路径哦， 没有默认搜索路径
fileids, 
encoding='utf8', 
word_tokenizer=<nltk.tokenize.simple.TabTokenizer object at 0x112751e10>, sent_tokenizer=RegexpTokenizer(pattern='n', gaps=True, discard_empty=True, flags=56), 
para_block_reader=<function read_blankline_block at 0x112693d10>)
```

OK， 了解。来一个实例

```python
dep_reader = DependencyCorpusReader('nltk_data/corpora/dependency_treebank', r'.*')
sents = dep_reader.parsed_sents() # 这样子得到一个列表
sent = sents[0] # 这样子得到 nltk.parse.dependencygraph.DependencyGraph
sent
# <DependencyGraph with 19 nodes> 默认就是这么显示了
sent.tree()
# Tree('will', [Tree('Vinken', ['Pierre', ',', Tree('old', [Tree('years', ['61'])]), ',']), Tree('join', [Tree('board', ['the']), Tree('as', [Tree('director', ['a', 'nonexecutive'])]), Tree('Nov.', ['29'])]), '.']) 可以得到 lexicalized 的 树表示
sent.to_conll(style:int) # 可以选择 Malt-Tab(3列 4列)， Conll-U(10列)
```

请注意， 如果语料是 conllu 的, 那么 Malt-Tab 格式提取 (form, xpos, head) 或者 (form, xpos, head, deprel) 而不是 upos.

这个真是太好了呢。不过从另一个角度来说说， nltk 果然有点慢， 而且吃内存很随便， 因为它把各种 View 都放在内存里面了， 丝毫不在意内存的使用， 所以大内存很重要啊。

但是它们仍然没有把 Conll 树库转换成为 shift reduce 序列的功能， 所以自己来写也还是一件非常可行的事情啦。

##  P.S. 

NLTK 还是一个很好玩的工具呢








 
