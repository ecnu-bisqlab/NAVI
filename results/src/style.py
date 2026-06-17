import matplotlib as mpl

from .config import DEFAULT_CONFIG


def set_pub_style() -> None:
    mpl.rcParams.update(DEFAULT_CONFIG.pub_rc_params)

__all__ = ['set_pub_style']
