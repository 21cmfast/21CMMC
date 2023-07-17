def test_emulator_install():
    from py21cmemu import Emulator
    from py21cmemu.get_emulator import get_emu_data
    get_emu_data()
    #emu = Emulator()

def test_emulator_runs():
    from py21cmemu import Emulator
    import numpy as np
    check = Emulator().predict(np.random.rand(9))
