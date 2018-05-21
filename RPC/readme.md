# RPC(Remote Procedure Call, 远程过程调用)

* 主要实现了Java client调用python server( 深度学习模型, 如 [Parlai](https://github.com/facebookresearch/ParlAI) )
* 主要的步骤为:
  * [使用Java调用Python服务器RPC](http://bbs.it-home.org/thread-68527-1-1.html), 用来远程启动深度学习模型
  * 修改模型输入输出格式为: 通过socket实现client-server通信<br>
    参考[Python与Java之间Socket通信](https://blog.csdn.net/ChenTianSaber/article/details/52274257?locationNum=4)
  * 实现client: [Java多线程](https://www.cnblogs.com/GarfieldEr007/p/5746362.html)
  
* 服务器采用腾讯云服务器: [完成电脑和服务器的SOCKET通信](http://bbs.qcloud.com/thread-21376-1-1.html)

## 使用Java调用Python服务器RPC

* 通过使用xmlrpc机制，让python程序和java程序之间RPC通信交互
* 配置java开发环境:
  * 下载jar包: [apache-xmlrpc-3.1.3-bin.tar.gz](https://archive.apache.org/dist/ws/xmlrpc/binaries/apache-xmlrpc-3.1.3-bin.tar.gz)<br>
    无法下载试试度盘: 链接: https://pan.baidu.com/s/1Y1e3Zh90PHuKKMSckUuWMA 密码: afi8
  * 解压得到5个jar包
  * 在Java工程中添加jar包: 右键工程 -> 选择properties -> Java Build Path -> Libraries -> add External JARs...
  * 具体的作用和用法可以参考它的官方API文档：http://ws.apache.org/xmlrpc/apidocs/index.html
* Python建立RPC服务器或客户端的通用库：
```python

```
