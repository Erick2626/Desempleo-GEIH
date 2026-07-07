import logging
from model.config.core import PACKAGE_ROOT, config

# Version del paquete
with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()

logging.getLogger(config.app_config.package_name).addHandler(logging.NullHandler())
