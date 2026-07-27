import os

bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
threads = int(os.getenv("WEB_THREADS", "4"))
timeout = 60
accesslog = "-"
errorlog = "-"
capture_output = True
control_socket_disable = True
