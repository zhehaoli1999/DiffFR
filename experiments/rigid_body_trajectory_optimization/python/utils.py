green_head = "\x1b[1;30;92m"
yellow_head = "\x1b[1;30;93m"
red_head = "\x1b[1;30;91m"
cyan_head = "\x1b[1;30;96m"
color_tail = "\x1b[0m"

def print_green(*msg):
    print("\x1b[1;30;92m", *msg, "\x1b[0m")

def print_yellow(*msg):
    print("\x1b[1;30;93m", *msg, "\x1b[0m")

def print_red(*msg):
    print("\x1b[1;30;91m", *msg, "\x1b[0m")

def print_cyan(*msg):
    print("\x1b[1;30;96m", *msg, "\x1b[0m")

def print_warning(*msg):
    print_yellow(*msg)

def print_error(*msg):
    print_red(*msg)

def print_success(*msg):
    print_green(*msg)

def print_info(*msg):
    print_cyan(*msg)
