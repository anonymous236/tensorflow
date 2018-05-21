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
   # -*- coding:utf-8 -*-
   # server.py
   from SimpleXMLRPCServer import SimpleXMLRPCServer
   from SocketServer import ThreadingMixIn
   from xmlrpclib import ServerProxy
   import thread
   import os
   import sys

   class ThreadXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
       pass
   class RPCServer():
       def __init__(self, ip='127.0.0.1', port='8000'):
           self.ip = ip
           self.port = int(port)
           self.svr = None
       def start(self, func_lst):
           thread.start_new_thread(self.service, (func_lst, 0,))
       def resume_service(self, v1, v2):
           self.svr.serve_forever(poll_interval=0.001)
       def service(self, func_lst, v1):
           self.svr = ThreadXMLRPCServer((self.ip, self.port), allow_none=True)
           for func in func_lst:
               self.svr.register_function(func)
               self.svr.serve_forever(poll_interval=0.001)
       def activate(self):
           thread.start_new_thread(self.resume_service, (0, 0,))
       def shutdown(self):
           try:
               self.svr.shutdown()
           except Exception, e:
               print 'rpc_server shutdown:', str(e)

   class Logger(object):
       def __init__(self, filename="Default.log"):
           self.terminal = sys.stdout
           self.log = open(filename, "a")
       def write(self, message):
           self.terminal.write(message)
           self.log.write(message)
       def flush(self):
           pass

   class RPCClient():
       def __init__(self, ip='127.0.0.1', port='8000'):
           self.svr = ServerProxy('http://'+ip+':'+port+'/', allow_none=True, use_datetime=True)
       def get_svr(self):
           return self.svr
       def get_hello():
           return 'hello!'
       def run_parlai():
           print("\nbegin to start.\n")
           os.system("python3 /root/ParlAI/examples/eval_model.py -m local_human -t babi:Task1k:1 -dt valid")
           return 0
       if __name__ == "__main__":
           r = RPCServer('服务器地址', '8000')
           r.service([run_parlai], 0) #这里仅仅载入run_parlai函数
   ```
 * 启动server.py待用
   ```shell
   python server.py
   ```
 * Java client: Java调用Python服务器RPC
