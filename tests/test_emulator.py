def test_emulator_runs():
    from py21cmemu import Emulator
    import numpy as np
    emu = Emulator()
    emu.predict(np.random.rand(9))
