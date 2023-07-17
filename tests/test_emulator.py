import numpy as np

def test_emulator_install():
    from py21cmemu import Emulator
    emu = Emulator()

def test_emulator_runs():
    from py21cmemu import Emulator
    check = Emulator().predict(np.random.rand(9))
