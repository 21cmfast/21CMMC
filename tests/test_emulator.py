def test_emulator_install():
    from py21cmemu import Emulator
    from py21cmemu.get_emulator import get_emu_data
    get_emu_data()

    import tensorflow as tf
    from py21cmemu.config import CONFIG
    m = tf.keras.models.load_model(CONFIG.data_path / '21cmEMU'/'21cmEMU')
    #emu = Emulator()

def test_emulator_runs():
    from py21cmemu import Emulator
    import numpy as np
    check = Emulator().predict(np.random.rand(9))
