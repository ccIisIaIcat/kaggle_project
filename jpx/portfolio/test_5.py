
from threading import Thread
 
global_num = 0
 
def func1():
	global global_num
	for i in range(1000000):
		global_num += 1
	print('---------func1:global_num=%s--------'%global_num)
 
def func2():
	global global_num
	for i in range(1000000):
		global_num += 1
	print('--------fun2:global_num=%s'%global_num)
print('global_num=%s'%global_num)
 
# lock = Lock()
 
t1 = Thread(target=func1)
t1.start()
 
t2 = Thread(target=func2)
t2.start()

