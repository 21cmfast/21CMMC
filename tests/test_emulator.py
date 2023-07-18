def test_emulator_install():
    from py21cmemu import Emulator
    from py21cmemu.get_emulator import get_emu_data
    get_emu_data()

    import tensorflow as tf
    from py21cmemu.config import CONFIG
    import os 
    print(os.listdir(CONFIG.data_path / '21cmEMU'))
    print(os.listdir(CONFIG.data_path / '21cmEMU'/'21cmEMU'))

    m = tf.keras.models.load_model(CONFIG.data_path / '21cmEMU'/'21cmEMU')

    emu = Emulator()
