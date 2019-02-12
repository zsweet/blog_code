文件描述：

- `graph.py`:tensorboard中graph的语法demo
- `graph_multisteps`:tensorboard中记录graph执行情况demo,记录执行时cpu、memory的相关情况，便于debug
- `scalar_graph_histogram`:因为scalar和histogram的用法特别相似，并且graph作为基础的功能，在使用scalar或histogram时自动生成graph，因此融合了这三个模块做一个demo
- `log`文件夹：运行上述三个脚本文件生成的相关log存放在该文件夹，与脚本名称对应

运行方式：

download该文件夹(注意存放路径不要有中文)，cd到该tensorboard文件夹，然后执行:
`tensorboard --logdir=log`，然后再Chrome中打开提示的链接