---
layout: post
title: 优雅的参数解析
categories: parse
description: argparse, conffig parse, env parse, 各种parse
keywords: configure
---  

## 如何从程序外部获得参数

如何从程序外部获得参数，或者说如何配置程序，这是一个颇为麻烦的事情。对于一个函数而言，调用它的时候，参数写在括号里，传进去就是了。对于程序，最原始的做法就是 argc, argv, 这是标准的 C 的方法。而且就一切皆字符，要什么类型自己 cast 的 C 语言而言，就只有这些了，要更高级的轮子，自己造去。

那么我们来说说一个配置程序的库应该满足那些需求吧。

1. 能够定义 parser 或者模板，定义了能接受哪些参数， 并能够自动生成 `--help` 文件。
2. 有类型和默认值。
3. 能够从多个源读取配置，如命令行，配置文件，环境变量，并且能够定义优先级次序。

以及一些其他的可选的支持：

1. 支持配置文件字符串插值方法。
2. 配置文件支持注释。
3. 支持把解析到的结果重新保存，这对于深度学习实验结果记录很有帮助。

## python 世界的选择

那么来说一下实际的存在的一些库。

1. argparse 是一个还不错的命令行参数解析库。但是它的 bool 类型实在有点坑。以及虽然它支持从文件读取参数，但是不是一个合适的配置文件的方式，而是类似直接把文件 cat 到命令行里面。argparse 对于 bool 类型的推荐方式应该是 `--store_true` 和 `__store_false` 等 action 的方式来处理的，这对于配置文件不友好。

2. configparse 是纯存的配置文件库。不支持定义模板或 parser, 读取配置文件的结果完全取决于配置文件，程序对此没有感知，也不能生成合适的 help 信息。而且 config 默认都是字符串类型，需要自己 cast, 而且不使用 `__getattr__` 语法而是 `__getitem__` 语法，比较繁琐。

3. configargparse 算是做了融合。最终还是一个 argparse。其实从程序配置的角度来说，还是 argparse 合适。类似于 boost 的 program option, 这才是真正的程序选项。 configparse 系列都只是变向读取字典而已。对于我个人而言，我觉得 configargparse 是一个很不错的库了。

4. click, fire 之类的，感觉侵入性太强，不喜欢。confuse 和 configparse 系列感觉差不多，还是有点繁琐。

实践中看到的有 ESPNet 使用 configargparse, 说明 configargparse 还是经历了实战的考验。以及 fairseq 中有一个技巧。下面解释一下 fairseq 中的技巧。

## fairseq 的技巧

fairseq 下的 train.py 文件的帮助信息是。

```bash
(scientific) [23:00]clementine@SUSE:fairseq> python train.py --help
usage: train.py [-h] [--no-progress-bar] [--log-interval N]
                [--log-format {json,none,simple,tqdm}]
                [--tensorboard-logdir DIR] [--tbmf-wrapper] [--seed N] [--cpu]
                [--fp16] [--memory-efficient-fp16]
                [--fp16-init-scale FP16_INIT_SCALE]
                [--fp16-scale-window FP16_SCALE_WINDOW]
                [--fp16-scale-tolerance FP16_SCALE_TOLERANCE]
                [--min-loss-scale D]
                [--threshold-loss-scale THRESHOLD_LOSS_SCALE]
                [--user-dir USER_DIR] [--empty-cache-freq EMPTY_CACHE_FREQ]
                [--criterion {sentence_prediction,binary_cross_entropy,sentence_ranking,cross_entropy,nat_loss,label_smoothed_cross_entropy,label_smoothed_cross_entropy_with_alignment,composite_loss,legacy_masked_lm_loss,adaptive_loss,masked_lm}]
                [--tokenizer {moses,space,nltk}]
                [--bpe {fastbpe,sentencepiece,bert,subword_nmt,gpt2}]
                [--optimizer {sgd,adagrad,adafactor,adadelta,adam,adamax,nag}]
                [--lr-scheduler {fixed,inverse_sqrt,reduce_lr_on_plateau,tri_stage,triangular,cosine,polynomial_decay}]
                [--task TASK] [--num-workers N]
                [--skip-invalid-size-inputs-valid-test] [--max-tokens N]
                [--max-sentences N] [--required-batch-size-multiple N]
                [--dataset-impl FORMAT] [--train-subset SPLIT]
                [--valid-subset SPLIT] [--validate-interval N]
                [--fixed-validation-seed N] [--disable-validation]
                [--max-tokens-valid N] [--max-sentences-valid N]
                [--curriculum N] [--distributed-world-size N]
                [--distributed-rank DISTRIBUTED_RANK]
                [--distributed-backend DISTRIBUTED_BACKEND]
                [--distributed-init-method DISTRIBUTED_INIT_METHOD]
                [--distributed-port DISTRIBUTED_PORT] [--device-id DEVICE_ID]
                [--distributed-no-spawn] [--ddp-backend {c10d,no_c10d}]
                [--bucket-cap-mb MB] [--fix-batches-to-gpus]
                [--find-unused-parameters] [--fast-stat-sync] --arch ARCH
                [--max-epoch N] [--max-update N] [--clip-norm NORM]
                [--sentence-avg] [--update-freq N1,N2,...,N_K]
                [--lr LR_1,LR_2,...,LR_N] [--min-lr LR] [--use-bmuf]
                [--save-dir DIR] [--restore-file RESTORE_FILE]
                [--reset-dataloader] [--reset-lr-scheduler] [--reset-meters]
                [--reset-optimizer] [--optimizer-overrides DICT]
                [--save-interval N] [--save-interval-updates N]
                [--keep-interval-updates N] [--keep-last-epochs N] [--no-save]
                [--no-epoch-checkpoints] [--no-last-checkpoints]
                [--no-save-optimizer-state]
                [--best-checkpoint-metric BEST_CHECKPOINT_METRIC]
                [--maximize-best-checkpoint-metric]

optional arguments:
  -h, --help            show this help message and exit
  --no-progress-bar     disable progress bar
  --log-interval N      log progress every N batches (when progress bar is
                        disabled)
  --log-format {json,none,simple,tqdm}
                        log format to use
  --tensorboard-logdir DIR
                        path to save logs for tensorboard, should match
                        --logdir of running tensorboard (default: no
                        tensorboard logging)
  --tbmf-wrapper        [FB only]
  --seed N              pseudo random number generator seed
  --cpu                 use CPU instead of CUDA
  --fp16                use FP16
  --memory-efficient-fp16
                        use a memory-efficient version of FP16 training;
                        implies --fp16
  --fp16-init-scale FP16_INIT_SCALE
                        default FP16 loss scale
  --fp16-scale-window FP16_SCALE_WINDOW
                        number of updates before increasing loss scale
  --fp16-scale-tolerance FP16_SCALE_TOLERANCE
                        pct of updates that can overflow before decreasing the
                        loss scale
  --min-loss-scale D    minimum FP16 loss scale, after which training is
                        stopped
  --threshold-loss-scale THRESHOLD_LOSS_SCALE
                        threshold FP16 loss scale from below
  --user-dir USER_DIR   path to a python module containing custom extensions
                        (tasks and/or architectures)
  --empty-cache-freq EMPTY_CACHE_FREQ
                        how often to clear the PyTorch CUDA cache (0 to
                        disable)
  --criterion {sentence_prediction,binary_cross_entropy,sentence_ranking,cross_entropy,nat_loss,label_smoothed_cross_entropy,label_smoothed_cross_entropy_with_alignment,composite_loss,legacy_masked_lm_loss,adaptive_loss,masked_lm}
  --tokenizer {moses,space,nltk}
  --bpe {fastbpe,sentencepiece,bert,subword_nmt,gpt2}
  --optimizer {sgd,adagrad,adafactor,adadelta,adam,adamax,nag}
  --lr-scheduler {fixed,inverse_sqrt,reduce_lr_on_plateau,tri_stage,triangular,cosine,polynomial_decay}
  --task TASK           task
  --dataset-impl FORMAT
                        output dataset implementation

Dataset and data loading:
  --num-workers N       how many subprocesses to use for data loading
  --skip-invalid-size-inputs-valid-test
                        ignore too long or too short lines in valid and test
                        set
  --max-tokens N        maximum number of tokens in a batch
  --max-sentences N, --batch-size N
                        maximum number of sentences in a batch
  --required-batch-size-multiple N
                        batch size will be a multiplier of this value
  --train-subset SPLIT  data subset to use for training (train, valid, test)
  --valid-subset SPLIT  comma separated list of data subsets to use for
                        validation (train, valid, valid1, test, test1)
  --validate-interval N
                        validate every N epochs
  --fixed-validation-seed N
                        specified random seed for validation
  --disable-validation  disable validation
  --max-tokens-valid N  maximum number of tokens in a validation batch
                        (defaults to --max-tokens)
  --max-sentences-valid N
                        maximum number of sentences in a validation batch
                        (defaults to --max-sentences)
  --curriculum N        don't shuffle batches for first N epochs

Distributed training:
  --distributed-world-size N
                        total number of GPUs across all nodes (default: all
                        visible GPUs)
  --distributed-rank DISTRIBUTED_RANK
                        rank of the current worker
  --distributed-backend DISTRIBUTED_BACKEND
                        distributed backend
  --distributed-init-method DISTRIBUTED_INIT_METHOD
                        typically tcp://hostname:port that will be used to
                        establish initial connetion
  --distributed-port DISTRIBUTED_PORT
                        port number (not required if using --distributed-init-
                        method)
  --device-id DEVICE_ID, --local_rank DEVICE_ID
                        which GPU to use (usually configured automatically)
  --distributed-no-spawn
                        do not spawn multiple processes even if multiple GPUs
                        are visible
  --ddp-backend {c10d,no_c10d}
                        DistributedDataParallel backend
  --bucket-cap-mb MB    bucket size for reduction
  --fix-batches-to-gpus
                        don't shuffle batches between GPUs; this reduces
                        overall randomness and may affect precision but avoids
                        the cost of re-reading the data
  --find-unused-parameters
                        disable unused parameter detection (not applicable to
                        no_c10d ddp-backend
  --fast-stat-sync      Enable fast sync of stats between nodes, this
                        hardcodes to sync only some default stats from
                        logging_output.

Model configuration:
  --arch ARCH, -a ARCH  Model Architecture

Optimization:
  --max-epoch N, --me N
                        force stop training at specified epoch
  --max-update N, --mu N
                        force stop training at specified update
  --clip-norm NORM      clip threshold of gradients
  --sentence-avg        normalize gradients by the number of sentences in a
                        batch (default is to normalize by number of tokens)
  --update-freq N1,N2,...,N_K
                        update parameters every N_i batches, when in epoch i
  --lr LR_1,LR_2,...,LR_N, --learning-rate LR_1,LR_2,...,LR_N
                        learning rate for the first N epochs; all epochs >N
                        using LR_N (note: this may be interpreted differently
                        depending on --lr-scheduler)
  --min-lr LR           stop training when the learning rate reaches this
                        minimum
  --use-bmuf            specify global optimizer for syncing models on
                        different GPUs/shards

Checkpointing:
  --save-dir DIR        path to save checkpoints
  --restore-file RESTORE_FILE
                        filename from which to load checkpoint (default:
                        <save-dir>/checkpoint_last.pt
  --reset-dataloader    if set, does not reload dataloader state from the
                        checkpoint
  --reset-lr-scheduler  if set, does not load lr scheduler state from the
                        checkpoint
  --reset-meters        if set, does not load meters from the checkpoint
  --reset-optimizer     if set, does not load optimizer state from the
                        checkpoint
  --optimizer-overrides DICT
                        a dictionary used to override optimizer args when
                        loading a checkpoint
  --save-interval N     save a checkpoint every N epochs
  --save-interval-updates N
                        save a checkpoint (and validate) every N updates
  --keep-interval-updates N
                        keep the last N checkpoints saved with --save-
                        interval-updates
  --keep-last-epochs N  keep last N epoch checkpoints
  --no-save             don't save models or checkpoints
  --no-epoch-checkpoints
                        only store last and best checkpoints
  --no-last-checkpoints
                        don't store last checkpoints
  --no-save-optimizer-state
                        don't save optimizer-state as part of checkpoint
  --best-checkpoint-metric BEST_CHECKPOINT_METRIC
                        metric to use for saving "best" checkpoints
  --maximize-best-checkpoint-metric
                        select the largest metric value for saving "best"
                        checkpoints
```

相当的惊人。而当你去看 `train.py` 并没有发现一堆面条代码 `parser.add_argument` 之类的。而只有 

```python
def cli_main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
```

这说明， parser 的定义放到了别处。再跟着跑过去一看，发现类似这样的函数， 里面定义了一套 parser.

```python
def get_training_parser(default_task='translation'):
    parser = get_parser('Trainer', default_task)
    add_dataset_args(parser, train=True)
    add_distributed_training_args(parser)
    add_model_args(parser)
    add_optimization_args(parser)
    add_checkpoint_args(parser)
    return parser
```

模式很明显， 给定 task， task 会产生一个比较基础的 parser, 然后各个部分往这个 parser 里面添加 key, 比如 dataset 相关的， task 相关的， 保存加载相关的， 模型相关的， optimizer 相关的， 分布式相关的， 训练过程相关的。一般来说一个深度学习程序所需要的配置也就这些，但是要实现可复用，也是需要好好设计。比如如果一个程序没有实现分布式训练，那么分布式的参数就不应该出现，比如每个模型可能有不同的可配置参数，每个数据预处理方式可能有不同的可配置参数，怎么能够实现复用呢？ 实践中，因为我们常常放弃考虑可复用性，我们不是在做 library 或者 toolkit, 而只是写一个 recipe， 训练完，跑个分就完事了，甚至不考虑以后怎么加载，所以实际上不怎么设计，想到哪就写到哪。所以一般也是一个工程一个 parser, 写在 main 文件里。最后发现 main 文件里面一长串的面条代码。

所以我的感觉是，设计一些抽象接口，然后每个模块去实现自己的对 parser 的操作，去添加自己需要的 key 是一个比较好的实践。比如说 fairseq 中的模型使用了一个 `add_arguments` 类方法去添加自己需要的 key. 距离如下。

```python
class RobertaModel(FairseqLanguageModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-layers', type=int, metavar='L',
                            help='num encoder layers')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A',
                            help='num encoder attention heads')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN')
        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--load-checkpoint-heads', action='store_true',
                            help='(re-)register and load heads when loading checkpoints')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
```

这样就可以很清楚地知道， 这个模型会用到这些参数，所以它自己注册这些 Key， 并且注册和使用的距离很近， 每个参数是什么意义很清楚， 所以用户不会读代码跳来跳去。不过值得注意的是：

1. faieseq 有 module 和 model 的抽象层级之分， module 是通用性很强的模块， 它们是被 model 调用的， 不需要从程序外部获取参数， 所以它们不需要这样做。而 model 是比较完整的包装， 配置好就能直接用了，而且一般情况下是作为一个任务的重要的可配置项的来源，它使用这种方法是合理的。
2. 需要始终保证所有的 `add_args` 方法作用到同一个 parser 上。
3. argparse 的 key 是没有层级结构的，即使它有一个 `argument_group` 也只是为了区别一下，为了好看，如何避免名字冲突？没有办法，要么就添加前缀，但是这样会使得 key 变得很长，而且还有使用 `-` 还是 `_` 连接单词的问题， ini 配置文件支持 section, yaml 和 json 之类的也支持更灵活的结构， 但是对于命令行参数来说， 这明显太过分。

所以呢， 通过配置文件来传参数还是比较使用的方法。至于 key 很多， key 很长， 这个似乎是没有办法的事， 谁叫训模型这件事有这么多的可配置项呢。



