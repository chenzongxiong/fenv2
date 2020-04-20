import multiprocessing
import threading
import constants


class ProcessPool(object):
    _pool = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._pool:
            with cls._lock:
                if not cls._pool:
                    cls._pool = multiprocessing.Pool(constants.CPU_COUNTS)

        return cls._pool


_pool = multiprocessing.Pool(constants.CPU_COUNTS)
