r1说从sft的model开始train from scratch的话，上限不高。
对此我的看法是：因为sft严重的finetune了model的参数，在这个过拟合的参数上进行rl，如果困于exploit，会导致rl不起来，如果可以大幅地explore的话，是不是会打破这一现象？


对此，我的计划是：
model-a: 跑一版正常从base model开始训的rl
model-b: 对一个base model进行sft，然后在这个基础上进行rl（想用reinforce++）

比较model-a和model-b的效果差异，（默认是有差异的），开始调整如何可以more explore？如何在rl训练过程中体现more explore?

Entropy? 
熵高的话，对所有可能的动作(token)分配更均匀的概率-（探索）
熵低的话，策略集中在少数高概率动作上-（利用）
