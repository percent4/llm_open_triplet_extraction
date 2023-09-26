本项目使用大语言模型（LLM）进行开放领域三元组抽取。

基座模型为`Baichuan-7B`，训练框架采用`Firefly`.

### 数据集

参考`data/spo.json`，三元组数量分布图如下：

![](https://raw.githubusercontent.com/percent4/llm_open_triplet_extraction/main/data/triples_distribution.png)

### 模型训练

### 测试案例

#### 例子1

来源网址：[https://www.chinanews.com/cj/2023/09-25/10083719.shtml](https://www.chinanews.com/cj/2023/09-25/10083719.shtml)

抽取结果：

![](https://s2.loli.net/2023/09/27/NMKr6adWeQh39XL.png)

图谱展示：

![](https://s2.loli.net/2023/09/27/Xj1DHU2d7pLEKfJ.png)

#### 例子2

来源网址：[https://www.jjxw.cn/xinwen/jjsz/202309/t20230926_6225481.html](https://www.jjxw.cn/xinwen/jjsz/202309/t20230926_6225481.html)

抽取结果：

![](https://s2.loli.net/2023/09/27/nEchIxVk6MAXf8S.png)

图谱展示：

![](https://s2.loli.net/2023/09/27/7sxWpgQeF9JXwAT.png)

