def test_emulator_install():
    from py21cmemu import Emulator
    from py21cmemu.config import CONFIG

    import tensorflow as tf
    print(tf.random.normal((1,10)))
    print(CONFIG.data_path)
    import os
    print(os.listdir(CONFIG.data_path))
    emu = Emulator()

def test_emulator_runs():
    from py21cmemu import Emulator
    import numpy as np
    check = Emulator().predict(np.random.rand(9))
