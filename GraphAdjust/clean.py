from sysv_ipc import *
import os
import subprocess
import signal
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
key =  ftok("/mnt/2/sjm_env/ws_moveit/src/moveit_tutorials/doc/motion_planning_api/fraction",123)
queue = MessageQueue(key)
queue.remove()

pids = get_process_id('tuto')
for pid in pids:
    os.kill(pid,signal.SIGKILL)