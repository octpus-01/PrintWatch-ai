import time
import os

LOGDIR = "log.txt"

def log_write(what):
    what = str(what)
    if os.path.exists(LOGDIR):
        print("log","logfile exist")
    
    with open(LOGDIR,"a") as file:
        file.write("\n" + what)

def log_info(who, message):
    info = "[info]" +" "+ time.asctime() + who + message
    print(info)
    log_write(info)

def log_error(who, error):
    error_out = "[error]" +" "+ time.asctime() + who + error
    print(error_out)
    log_write(error_out)



if __name__ == "__main__":
    log_info("hello","shit!")
