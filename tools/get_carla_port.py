import subprocess
import random

def get_free_port():
    """
    Returns a free port.
    """

    # get random in between 2000 and 3000 divisble by 5
    port = random.randint(40000, 60000)
    #port = 2000
    port_free = False

    while not port_free:
        try:
            pid = int(subprocess.check_output(f"lsof -t -i :{port} -s TCP:LISTEN", shell=True,).decode("utf-8"))
            pid = int(subprocess.check_output(f"lsof -t -i :{port+8000} -s TCP:LISTEN", shell=True,).decode("utf-8"))
            # print(f'Port {port} is in use by PID {pid}')
            port = random.randint(4000, 6000)

        except subprocess.CalledProcessError:
            port_free = True
            print(port)
            # print(f'Port {port} is free')

get_free_port()
