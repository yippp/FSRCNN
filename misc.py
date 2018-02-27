import sys
import time

TOTAL_BAR_LENGTH = 50
LAST_T = time.time()
BEGIN_T = LAST_T


def progress_bar(current, total, msg=None):
    global LAST_T, BEGIN_T
    if current == 0:
        BEGIN_T = time.time()  # Reset for new bar.

    sys.stdout.write('%d batches' % total)

    current_time = time.time()
    LAST_T = current_time
    total_time = current_time - BEGIN_T

    time_used = ' '
    time_used += 'Time: {:.3f}s'.format(total_time)
    if msg:
        time_used += ' | ' + msg

    msg = time_used
    sys.stdout.write(msg)

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

