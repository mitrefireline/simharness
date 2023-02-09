import os
from importlib.metadata import version

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["SDL_AUDIODRIVER"] = "dsp"

__version__ = version("simharness2")

del version
