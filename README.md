# paper
已阅读过的gnn 相关或不相关 paper 的总结，侧重NLP领域

# GNN

## GCN

### GCN

手写数字识别
spetial GCN结果优于spectral GCN
但是，后续一阶spectral GCN比较多

- Spectral Networks and Deep Locally ConnectedNetworks on Graphs.pdf
- spetial
- spectral

### GCN kipf

GCN最著名
边的权重，固定不可变
节点分类任务
简化了原始的谱卷积，简化为1阶

- SEMI-SUPERVISEDCLASSIFICATION WITHGRAPHCONVOLUTIONALNETWORKS.pdf

### GCN  co-training random walk

证明GCN过平滑的原理
与random walk协同训练减缓过平滑，治标不治本
公式计算迭代止整图传播均匀，不依赖验证集

- Deeper Insights Into Graph Convolutional Networks for Semi-Supervised Learning.pdf

### Predict then Propagate: Graph Neural Networks meet Personalized PageRank

GCN与pagerank结合
随机游走的距离过长，其因起点导致的差异会越来越小。同时GCN over-smooth
pagerank 算法进行全局更新

### text

- Text GCN

  文本分类(word node + doc node)
  依靠词互信息构建word node的邻接矩阵
  tf-idf作为 doc node与word node之间边的权重
  
  problem
  丢失词的位置信息
  对于情感分类不适用
  并不能对句子进行深层表征
  比较适合乱序给词就可拆除意思的文本

	- Graph Convolutional Networks for Text Classification.pdf

- HR-DGCNN

	- 子主题 1

	  每一个document 构建一张图
	  每个docment取前n个节点用于document 与document之间对齐
	  根据词共现矩阵提取子图，也就是重要节点，BFS生成全部的子图（填充子图大小到指定大小）
	  图对齐之后使用多层CNN进行卷积
	  对比多层CNN方法存在提升

### 子领域

- dialogue

	- DialogueGCN

	  多尺度CNN捕捉utterance特征
	  依照对话顺序，将所有话语GRU捕捉顺序
	  GCN以utterance作为node,speaker的说话者与接听者作为边的类型，并且包含双向,(邻接矩阵是否包含并不知道)
	  结合顺序Bi-GRU与speaker level的GCN综合对话语进行分类。
	  problem:
	  缺少speaker自己的特质，并且在初始时对于speaker与speaker之间的关系不同
	  边的类型：
	  speaker relation
	  时序 temporal
	  使用多模态数据集，只使用文本处理

		- DialogueGCN A Graph Convolutional Neural Network forEmotion Recognition in Conversation.pdf

	- ConGCN

	  MELD 老友记的多方对话数据集
	  语料库构建一整张对话图
	  图中包含speaker utterance 节点
	  同一utterance 节点相连
	  utterance 与speaker 节点相连
	  第一个使用图模型做对话情感识别

## GAT

仅依靠两个节点信息

### graph attention network

- GRAPHATTENTIONNETWORK.pdf

### Graph Star Net

inductive framework
建立Star node 捕获全局
可以处理 node 、edge、graph 3种分类任务
在情感识别任务中,提出了topic-sentiment (对IMDB种，每一个主题的影评作为doc node ，并与每一个topic节点相连)
node 分类：不能优于transductive实验，但是优于inductive 实验。
aggregate:类似GAT
文本分类任务:
node为词，
词node之间的边依赖是否出现在同一窗口中,即使词相同但位置不同也被视为不同的词。
基于此法，不使用动态图，不利于语义图的刻画
problem
1任务1模型，
transductive实验： node分类结果没有GAT GCN好。
一些实验需要的是局部信息，不是全局信息。

- Graph Star Net for Generalized Multi-Task Learning.pdf

### dual graph convolution network

加强了边的表示，并且可以将边分类问题转为节点分类问题

- nlp

	- knowledge graph

		- RDGCN

		  对于两个异构的知识图谱进行对齐
		  已有部分对齐实体
		  loss计算依靠此部分对齐实体
		  https://arxiv.gg363.site/abs/1908.08210

### 子领域

- HetGNN

  异构图
  
  任务：链接预测、推荐
  
  问题： 图表示,考虑不同类型的节点
  
  模型结构：
  
  random walk节点采样
  
  1、融合节点内不同类型属性 bilstm
  
  2、抽取邻接点 bilstm
  
  3、融合不同类型的节点 attention
  
  结果：优于baseline
  
  KDD

- 文本生成

	- Text Generation from Knowledge Graphs with Graph Transformers

	  任务：摘要生成
	  在生成过程融入了知识
	  transformer结构，在attention步骤使用知识
	  AGENDA数据集创建，摘要，与知识图谱

## Graphsage

问题：传统GCN等方法属于转导，不能泛化到未知节点，以及其他图
方法：训练聚合函数，训练每次输入只输入节点、邻接点以及不相关的节点计算loss
结果：优于GCN
问题：测试集中的结构，不能用于改进训练集
要求：必须保持节点类型对齐，以及边的类型对齐

## 全连接图

### Graph Transformer

ICLR被拒
全连接图
attention确认节点之间的边
可以自动构建图
长距离依赖
任务:
few-shot
医学影像
problem:
与attenton相似
对比的模型都很弱
只是单纯构建全连接图的意义不大，必须在不同图之间建立信息链，利用全局信息
全连接图的噪音过大，构建一张准确的图很难。

- GRAPHTRANSFORM.pdf

## factor graph

### factor graph network

被拒，不知道去哪
捕捉高阶依赖
factor node 可连接多个node
双向传播:
factor->node 
node->factor
对于最大后验概率计算，已知正确的势能函数（factor 计算函数）效果很好。不过dataset3的结果不如LP relaxation 是因为dataset3不确定性太大，而factor graph  network并不适应
Point Cloud则不行

- Factor+Graph+Neural+Network.pdf

### factor attention graph

attention对任意数量的数据没有要求
graph:node*5(图像、标题、历史问题、历史答案、问题、答案)
通过attention实现factor的表现形式。主要self-attention factor 以及节点之间交互的attention

- Factor Graph Attention.pdf

### expressGNN

knowledge  graph 进行补全，发现未知的知识
factor node表示关系类型
实体使用node表示
GNN不知道哪种，未开源

- Can Graph Neural Networks Help Logic Reasoning.pdf

## 子领域

### 可视化问答

- Learning by Abstraction: The Neural State Machine

  针对可视化问答
  提出了构建语义概率图，
  将图像与文本均转为图中的语义概念
  使用顺序推理
  算法无创新
  优点：
  强调构建世界模型

### 知识图谱

- Estimating Node Importance in Knowledge Graphs Using Graph Neural Networks

  知识谱上节点重要性估计
  考虑了以下5点信息：
  Neighborhood Awareness
  Making Use of Predicates.
  Centrality Awareness
  Utilizing Input Importance Scores.
  Flexible Adaptation
  
  GNN的架构
  结果高于GAT但是应该不是SOTA

## latent variable

此系列方法多有点辅助任务的感觉，很像预训练转为有监督的过程。或者认为取出了中间变量

### Scalable Temporal Latent Space Inference for Link Prediction in Dynamic Social Networks

问题：获取动态的graph embedding 来进行link pprediction
任务：社交网络链路预测（认为节点的变化应该很稳定，不会突然发生剧烈变化）
方法：
改进BCGD
	只依靠前一时刻获取下一时刻的表示，并且使用时间正则loss避免前后时间相差过大。在进行更新时只使用邻接点进行更新，避免cost较大(没看懂？？？)
结果：看着可以

### VAE

- Neural Relational Inference for Interacting Systems

  任务：在不给定图结构的情况下，预测图的结构，以及节点下一时刻的状态。粒子仿真系统 （输入：不同时刻节点的状态）
  方法：借鉴VAE思想，利用z来表示节点之间固有的图结构，应该指的一种客观规律，不随时间变化而改变。根据z 以及各时刻节点的表示，动态生成边的表示，再生成下一时刻节点的表示。
  实验结果：
  效果不错，与lstm做对比，也许是第一个在这个方向上做的
  想法：可以用做对话生成，使用无监督的方法构建图

### Dynamic Graph Representation Learning via Self-Attention Networks

问题：学习dynamic graph的 node representation 来预测边。以往都是基于Markov只考虑前一时刻的状态，本文self-attention自由结合之前时刻，任意选取。
方法：使用structure self-attention 来捕捉某一个时刻的图结构。使用temporal self-attention 来捕获当前时刻之前所有时刻的同一个节点的变化
数据集：社交网络，以及yelp 和ML影评
结果： 动态网络没有静态的表现好
启发：将图结构用于之前的时刻

## domain

### nlp

- dialogue

### cv

### multi-model

- dialogue
- 可视化问答

### multitask

## 数据集

### node classification

- Cora
- Citeseer
- PubMed 医学

### graph classification

- Enzymes
- D&D
- Proteins
- Mutag

### text classification

- 20ng
- R8
- R52
- Obsumed
- MR 影评，情感

## Graph Embedding

### Large-Scale Hierarchical Text Classification with Recursively Regularized Deep Graph-CNN

## random walk

### GraphRNA

节点带有attributes
在节点与属性构成的二分图
使用random walk 在二分图上下部分进行游走表示，node-attribute-node-attribute  node走的是节点的邻接点
walk结果输入GRU+pooling得到图节点表示
问题:attribute如此融合有点粗暴，因为attribute 与 node表示可能不在一个空间

## GNN分类

### task

- classification

	- node classification

		- basic node

		  最基础的分类，例如nlp的词节点

		- cluster

		  以一个节点表示一群节点(community)

			- factor graph???
			- hypergraph???

	- edge classification
	- graph classification

- emebdding

	- node
	- edge

### structure

- 大小

	- 大图：1

		- 多为transductive learning

	- 小图：多

- 同异构

	- 异构heterogeneous graph

		- 问题

			- 图与图中参数无法迁移

		- 解决方法

			- 所有图拼接在一起，转为1张图

	-  同构isomorphic graph

- depth???
- dual

  两张图，以另一张图的顶点为边

	- 子主题 1

### algorithm

- aggregate

	- Locality

		- GCN

	- global

		- Pagerank

	- Sequentiality

- depth
- 是否是动态

	- 子主题 1
	- dynamic

- 类型

	- transductive
	- inductive

## 图的结构不变（可以少边，不可以多边） 如果边的权重为0，则默认无边。

*XMind: ZEN - Trial Version*
