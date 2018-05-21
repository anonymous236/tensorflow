package com.rpc;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.net.MalformedURLException;
import java.net.Socket;
import java.net.URL;
import java.net.UnknownHostException;
import java.rmi.ConnectException;
import java.util.Scanner;

import org.apache.xmlrpc.XmlRpcException;
import org.apache.xmlrpc.client.XmlRpcClient;
import org.apache.xmlrpc.client.XmlRpcClientConfigImpl;
import org.apache.xmlrpc.client.XmlRpcHttpTransportException;
/**
 * 这是Java client和python server通信的相关代码
 * @author Z
 */

class Example {
	/**
	 * 这是Java client连接python server的教学样例
	 */
	public void e_run_py() throws MalformedURLException {
		// TODO Auto-generated method stub
		XmlRpcClientConfigImpl config = new XmlRpcClientConfigImpl();
		//config.setServerURL(new URL("http://192.168.77.89:8888/RPC2"));
		//URL是python服务端的SimpleXMLRPCServer(("192.168.77.89", 8888))，注意http和/RPC2
		config.setServerURL(new URL("http://server ip:8001/RPC2"));
		XmlRpcClient client = new XmlRpcClient();
		client.setConfig(config);
		Object[] params = null;
		try {
			// 返回的结果是字符串类型，强制转换res为String类型
			String res = (String) client.execute("run_parlai", params);
			System.out.println(res);
		} catch (XmlRpcException e11) {
			e11.printStackTrace();
		}
	}
	
	/**
	 * 这是Java client和python server进行socket通信的教学样例
	 */
	public void e_socket2py() {
		try {
			Socket socket = new Socket("server ip",8001);
			//获取输出流，向服务器端发送信息
			OutputStream os=socket.getOutputStream();//字节输出流
			PrintWriter pw=new PrintWriter(os);//将输出流包装为打印流
			pw.write("我是Java服务器");
			pw.flush();
			socket.shutdownOutput();//关闭输出流
			
			InputStream is=socket.getInputStream();
			BufferedReader in = new BufferedReader(new InputStreamReader(is));
			String info=null;
			while((info=in.readLine())!=null){
			System.out.println("我是客户端，Python服务器说："+info);
		}
		is.close();
		in.close();
		socket.close();
		} catch (UnknownHostException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
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
			Socket socket = new Socket("server ip",8061);
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
//          TODO Auto-generated catch block
//			e.printStackTrace();
		} catch (IOException e) {
			System.out.println("Connection Error.");
//			e.printStackTrace();
		}
	}

	/**
	 * 这个是Java client和python server通信的实践，server运行的是Parlai问答模型
	 * 这一版本是从Java client远程开启server的python深度学习模型的程序，然后进行socket通信 
	 */
	public void run_py_socket() throws MalformedURLException, InterruptedException {
		// TODO Auto-generated method stub
		XmlRpcClientConfigImpl config = new XmlRpcClientConfigImpl();
		//config.setServerURL(new URL("http://192.168.77.89:8888/RPC2"));
		//URL是python服务端的SimpleXMLRPCServer(("192.168.77.89", 8888))，注意http和/RPC2
		config.setServerURL(new URL("http://server ip:8000/RPC2"));
		XmlRpcClient client = new XmlRpcClient();
		client.setConfig(config);
		Object[] params = null;
		try {
			// 返回的结果是字符串类型，强制转换res为String类型
			System.out.println("Starting program...");
			client.execute("run_parlai", params);
			System.out.println("Program completed.");
//			Thread.sleep(1000);
//			socket2py();
		} catch (XmlRpcException e) {
			System.out.println("Connection Error.");
//			e.printStackTrace();
		}
	}
}

class RunnableSocket extends Thread {
	private Thread t;
	private int lable;
	
	public RunnableSocket (int l) {
		lable = l;
	}
	
	public void run() {
		Example ex = new Example();
		if(lable == 0) {
			try {
				ex.run_py_socket();
			} catch (MalformedURLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		else {
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			ex.socket2py();
		}
	}
}

public class Socket2Py {
	public static void main(String args[])throws Exception, MalformedURLException, XmlRpcHttpTransportException {		
		RunnableSocket r0 = new RunnableSocket(0);
		r0.start();
		RunnableSocket r1 = new RunnableSocket(1);
		r1.start();
	}
}
