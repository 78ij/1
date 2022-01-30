import os
import subprocess
import shlex
import queue
import threading
import time
import struct
import signal
from sysv_ipc import *

# def t_read_stdout(process, queue):
#     """Read from stdout"""

#     while True:
#         output = str(process.stdout.readline())
#         if output.find('aaaaafraction') != -1:
#             q.put(output)
#         #print(1)
#         #output2 = str(process.stderr.readline())
#         #q.put(output2)

#     return
def get_process_id(name):
    """Return process ids found by (partial) name or regex.

    >>> get_process_id('kthreadd')
    [2]
    >>> get_process_id('watchdog')
    [10, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61]  # ymmv
    >>> get_process_id('non-existent process')
    []
    """
    child = subprocess.Popen(['pgrep', '-f', name], stdout=subprocess.PIPE, shell=False)
    response = child.communicate()[0]
    return [int(pid) for pid in response.split()]
command = 'source /mnt/2/sjm_env/ws_moveit/rosenv/bin/activate;source /mnt/2/sjm_env/ws_moveit/devel/setup.bash;roslaunch moveit_tutorials motion_planning_api_tutorial.launch my_args:="file=/mnt/2/sjm_env/ws_moveit/src/moveit_tutorials/doc/motion_planning_api/src/scene_descriptor"'

process = subprocess.Popen(command,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           bufsize=-1,
                           executable='/bin/bash',
                           shell=True,close_fds=True)

# q = queue.Queue()
# t_stdout = threading.Thread(target=t_read_stdout, args=(process, q))
# t_stdout.daemon = True
# #t_stdout.start()
key =  ftok("/mnt/2/sjm_env/ws_moveit/src/moveit_tutorials/doc/motion_planning_api/fraction",123)
while True:
    try:
        queue = MessageQueue(key)
    except:
        continue
    break
pid = get_process_id('/src/scene_descriptor __name:=motion_planning_api_tutorial __log:')[0]

while True:
    (message,t) = queue.receive()
    print(struct.unpack('<f', message)[0])
    os.kill(pid,signal.SIGCONT)
queue.remove()
