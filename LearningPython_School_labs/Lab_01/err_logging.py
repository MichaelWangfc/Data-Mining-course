# err_logging.py

import logging

def foo(s):
    try:
        return 100 / int(s)
    except Exception as e:
        logging.exception(e)

foo(0)
print('END')
