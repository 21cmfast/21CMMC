def test_emulator_runs():
    import numpy as np
    from py21cmemu import Emulator

    emu = Emulator()
    emu.predict(np.random.rand(9))


