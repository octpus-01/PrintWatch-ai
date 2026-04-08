import time

def log_info(who, message):
    info = "[info]" + time.asctime() + who + message

    print(info)

def log_error(who, error):
    error_out = "[error]" + time.asctime() + who + error
    print(error_out)



if __name__ == "__main__":
    log_info("hello","shit!")
