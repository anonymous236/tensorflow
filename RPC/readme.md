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
   ```java
   /**
    * 这个是Java client和python server通信的实践，server运行的是Parlai问答模型
    * 这一版本是从Java client远程开启server的python深度学习模型的程序，然后进行socket通信 
    */
   public void run_py_socket() throws MalformedURLException, InterruptedException {
       //TODO Auto-generated method stub
       XmlRpcClientConfigImpl config = new XmlRpcClientConfigImpl();
       //config.setServerURL(new URL("http://192.168.77.89:8888/RPC2"));
       //URL是python服务端的SimpleXMLRPCServer(("192.168.77.89", 8888))，注意http和/RPC2
       config.setServerURL(new URL("http://地址:8000/RPC2"));
       XmlRpcClient client = new XmlRpcClient();
       client.setConfig(config);
       Object[] params = null;
       try {
           System.out.println("Starting program...");
           client.execute("run_parlai", params);
           System.out.println("Program completed.");
       } catch (XmlRpcException e) {
           System.out.println("Connection Error.");
           //e.printStackTrace();
       }
   }
   ```

## Python与Java之间Socket通信

* [Python服务器](http://www.runoob.com/python/python-socket.html)
  ```python
  import socket
  
  # create socket
  sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
  # to solve the problem: "Address already in use"
  # so that the port will be released immediately when the socket done.
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR  , 1)
  # i set a new port 8061
  sock.bind(("server IP",8061))
  sock.listen(5)
  conn, addr = sock.accept()
  # eval_model is a Parlai function, change it with adding a param "conn"
  eval_model(opt, conn, print_parser=parser)
  conn.send(bytes("exit\n" ,encoding="utf8"))
  conn.close()
  sock.shutdown(socket.SHUT_RDWR)
  sock.close()
  ```
* 在Parlai模型内部, 修改模型的输入输出为socket通信
  ```python
  from parlai.core.agents import Agent
  from parlai.core.worlds import display_messages
  import socket

  class LocalHumanAgent(Agent):
      def __init__(self, opt, conn, shared=None):
          super().__init__(opt)
          self.id = 'localHuman'
          self.episodeDone = False
          try:
              self.conn = conn
          except:
              print("init socket error!")

          self.conn.settimeout(None)

      def observe(self, msg):
          print(display_messages([msg]))
          # send message to client
          self.conn.send(bytes(display_messages([msg]) + "\n",encoding="utf8"))
          print("\nsend finished.\n")

      def act(self):
          obs = self.observation
          reply = {}
          reply['id'] = self.getID()
          # reply_text = input("Enter Your Message: ")
          print("Enter Your Message: ")

          # send message to client.
          self.conn.send(bytes("Enter Your Message: ",encoding="utf8"))
          # mark the last message with "eof"
          self.conn.send(bytes("eof\n",encoding="utf8"))
          # receive message from client
          szBuf=self.conn.recv(1024)
          print("recv:"+str(szBuf,'gbk'))
          reply_text = str(szBuf, encoding='utf8')
          print(type(szBuf))

          reply_text = reply_text.replace('\\n', '\n')
          reply['episode_done'] = False
          if '[DONE]' in reply_text:
              reply['episode_done'] = True
              self.episodeDone = True
              reply_text = reply_text.replace('[DONE]', '')
          reply['text'] = reply_text
          return reply

      def episode_done(self):
          return self.episodeDone
  ```
* [Java客户端](https://blog.csdn.net/ChenTianSaber/article/details/52274257)
  ```java
  /**
   * 这个是Java client和python server通信的实践，server运行的是Parlai问答模型
   * 这一版本是在服务器开启程序的情况下直接进行socket通信
   */
  public void socket2py() {
      try {
          // 返回的结果是字符串类型，强制转换res为String类型
          //client.execute("run_parlai", params);
          int index;
          String info=null;
          System.out.println("Socket connecting...");
          Socket socket = new Socket("211.159.164.64",8061);
          System.out.println("Socket conncted.\n");
          while(true) {
              InputStream is=socket.getInputStream();
              BufferedReader in = new BufferedReader(new InputStreamReader(is));
              while((info=in.readLine())!=null){     
                  if((index = info.indexOf("eof")) != -1) {
                      System.out.print(info.substring(0, index));
                      break;
                  }
                  System.out.println(info);
              }
              if(info == null) {
                  break;
              }

              Scanner s = new Scanner(System.in);
              String str = null;
              str = s.next();
              OutputStream os=socket.getOutputStream();//字节输出流
              PrintWriter pw=new PrintWriter(os);//将输出流包装为打印流
              pw.write(str);
              pw.flush();            
          }
          socket.shutdownOutput();//关闭输出流
          socket.close();
      } catch (UnknownHostException e) {
          //TODO Auto-generated catch block
          //e.printStackTrace();
      } catch (IOException e) {
          System.out.println("Connection Error.");
          //e.printStackTrace();
      }
  }
  ```
  
  
