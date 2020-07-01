from yacs.config import CfgNode as CN
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger("service-ai-app")

_C = CN(new_allowed=True)