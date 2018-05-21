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
		#sys.stdout = Logger("cont.txt")
                print("\nbegin to start.\n")
		os.system("python3 /root/ParlAI/examples/eval_model.py -m local_human -t babi:Task1k:1 -dt valid")
	        print("\nrun!!!\n")
                return 0
        if __name__ == "__main__":
		r = RPCServer('server ip', '8000')
		r.service([run_parlai], 0) #这里仅仅载入run_parlai函数
