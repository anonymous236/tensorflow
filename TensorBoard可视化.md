# TensorBoard可视化
TensorBoard的工作方式是启动一个Web服务，该服务进程从TensorBoard程序执行所得的事件日志文件(event files)中读取概要(summary)数据，然后将数据在网页中绘制成可视化图标。
## TensorBoard安装
```
pip install tensorboard
```
## 启动TensorBoard
* 问题描述：在terminal中输入tensorboard时提示找不到命令
* 解决方法：启动的时候需要换一种方式
  * 输入 *pip show tensorflow* 或者 *pip show tensorflow-gpu* 找到tensorflow的安装路径
  * 再启动 *python 安装路径/tensorflow/tensorboard/tensorboard.py --logdir=文件夹路径*
