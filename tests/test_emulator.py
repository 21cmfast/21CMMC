def test_emulator_runs():
    import numpy as np
    emu = Emulator()
    emu.predict(np.random.rand(9))
