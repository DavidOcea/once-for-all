


注意，搜索“手动”看哪些需要手动修改（训练集batch_size最好==验证集的，不然可能会报错）因为有些写死在代码里了）
utils/distributed_run_manager/run_manager

20200428,第一次ooce for all 实验，from scratch,默认优化深度: 因为与训练模型无法加载，所以把teach(train_oaf_net)和预训练pretrained(progressive_shrinking)都屏蔽了
 horovodrun -np 16 -H 127.0.0.1:4 python train_ofa_net.py --phase 2