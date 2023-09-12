#1.文件结构
├── src（自行创建的文件夹）
│   ├── deep_q_network.py（DQN网络模型）
│   └── tetris.py（游戏环境，为了训练效率，未实现可视化界面）
│
├── tensorboard（自行创建的文件夹，存放训练过程中自动生成的log）
├── test.py（测试代码）
├── trained_models（自行创建的文件夹，存放训练保存的模型）
└── train.py（训练代码）

#2.进行训练
按1中创建文件夹并安放好文件后，运行train.py
注意必须预留空文件夹，否则报错！

#3.观察曲线
设1中文件结构的父目录为folder
####①执行到folder的cd指令
####②执行指令tensorboard --logdir tensorboard
####③在浏览器中打开自动弹出的网址
####④用tensorboard查看