import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
logging.basicConfig(level=logging.INFO, filename="logfile", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
