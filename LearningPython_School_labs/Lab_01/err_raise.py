# err_raise.py
def foo(s):
    n = int(s)
    if n==0:
        raise ('It is not a invalid value: %s' % s)
    return 10 / n

foo('0')

