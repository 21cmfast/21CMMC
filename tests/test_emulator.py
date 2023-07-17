def test_emulator_install():
    from py21cmemu import Emulator
    import tensorflow as tf
    print(tf.random.normal(10))
    emu = Emulator()

def test_emulator_runs():
    from py21cmemu import Emulator
    import numpy as np
    check = Emulator().predict(np.random.rand(9))
