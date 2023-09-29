本项目使用大语言模型（LLM）进行开放领域三元组抽取。

### 数据集

参考`data/spo.json`，三元组数量分布图如下：

![](https://raw.githubusercontent.com/percent4/llm_open_triplet_extraction/main/data/triples_distribution.png)

数据已上传至HuggingFace，网址为: [https://huggingface.co/datasets/jclian91/open_domain_triple_extraction](https://huggingface.co/datasets/jclian91/open_domain_triple_extraction)

### 模型训练

基座模型为`Baichuan2-13B-Base`，训练框架采用`Firefly`.

参数如下：

```json
{
    "output_dir": "output/firefly-baichuan2-13b-spo",
    "model_name_or_path": "/workspace/Baichuan2-13B-Base",
    "train_file": "./data/spo.jsonl",
    "num_train_epochs": 10,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1e-4,
    "max_seq_length": 550,
    "logging_steps": 100,
    "save_steps": 100,
    "save_total_limit": 1,
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 300,
    "lora_rank": 64,
    "lora_alpha": 16,
    "lora_dropout": 0.05,

    "gradient_checkpointing": true,
    "disable_tqdm": false,
    "optim": "paged_adamw_32bit",
    "seed": 42,
    "fp16": true,
    "report_to": "tensorboard",
    "dataloader_num_workers": 0,
    "save_strategy": "steps",
    "weight_decay": 0,
    "max_grad_norm": 0.3,
    "remove_unused_columns": false
}
```

### 模型测试

参考`evaluate`文件夹。

个人收集的来自各个网站的新闻、小说中的三元组，文件为`evaluate_data.xlsx`，前几行如下：

| 文本 | 真实三元组 | 来源   | 网址 |
|----|-------|------|----|
|新华社杭州9月24日电（记者姬烨、董意行）国际奥委会主席巴赫23日在杭州出席了第19届亚运会开幕式，他称赞这场开幕式是数字创新和人文风采的完美结合。|(新华社，记者，姬烨)(新华社，记者，董意行)(国际奥委会，主席，巴赫)| 新华网  |https://www.news.cn/sports/2023-09/24/c_1212274341.htm|
|2022年11月，法国总统马克龙访问泰国，受到泰国国王哇集拉隆功接见。希里婉瓦丽出现在父亲身边。|(法国，总统，马克龙)(泰国，国王，哇集拉隆功)| 网易新闻 |https://www.163.com/dy/article/IFDIJR03051283GO.html|
|“这位是红岸基地的雷志成政委。我是杨卫宁，基地的总工程师。离降落还有一个小时，你休息吧。”|（红岸基地，政委，雷志成）（基地，总工程师，杨卫宁）|鲲弩小说|https://www.kunnu.com/santi/26653.htm|

评估脚本为`model_eval.py`，评估结果如下：

f1: 0.84831, precision: 0.90419, recall: 0.79894: : 100it [04:12,  2.52s/it]

具体的评估结果可参考`pred.json`.

### 抽取结果可视化

参考`visualize`文件夹。

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

