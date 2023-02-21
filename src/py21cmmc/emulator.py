import numpy as np
import os
import tensorflow as tf
import py21cmfast as p21
import logging

# Logging Config
LOGGING_CONFIG = {}
logging_format = "[%(asctime)s] %(process)d-%(levelname)s "
logging_format += "%(module)s::%(funcName)s():l%(lineno)d: "
logging_format += "%(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO)
log = logging.getLogger()

class p21cmEMU():
    r"""
    This class allows the use to load an emulator and use it to obtain 21cmFAST summaries.
    """
    def __init__(self,
                 emu_path : str = None,
                 io_options : dict = None,
                     ):
        """
        Parameters
        ----------
        emu_path : str
            Path of the emulator folder location including the folder name e.g. ./folder/21cmEMU
            If not set, default is in the current directory i.e. output of pwd
        
        
        """
        log.info('Init emulator...')
        if emu_path is None:
            log.warning('No emulator path was supplied. Trying to find the emulator folder in the current working directory.')
            # Check all local directories
            dirs = []
            for i in os.listdir('.'):
                if os.path.isdir(i):
                    dirs.append(i)
            mask = ['emu' in i.lower() for i in dirs]
            if any(mask):
                emu_path = np.array(dirs)[mask][0]
                log.info('Emulator path found: %s' % emu_path)
            else:
                raise ValueError('Cannot find the emulator folder. Please provide the path to it with emu_path keyword.')

        try:
            emu = tf.keras.models.load_model(emu_path, compile = False)
            log.info('Loaded the emulator successfully!')
        except OSError as e:
            raise ValueError('Wrong file path to emulator:', emu_path,' Remember to include the emulator folder name in the path.', e)

        self.model = emu
        self.io_options = io_options
        log.info('Find emulator_constants npz file.')
        if len(os.path.dirname(emu_path)) > 1:
            all_emulator_numbers = np.load(os.path.dirname(emu_path) + '/emulator_constants.npz')
        else:
            all_emulator_numbers = np.load('emulator_constants.npz')
        log.info('Successfully loaded emulator_constants npz file.')
        self.zs = all_emulator_numbers['zs']
        self.limits = all_emulator_numbers['limits']
        self.zs_cut = self.zs[:60]
        self.ks_cut = all_emulator_numbers['ks'][1:-3]
        self.PS_mean = all_emulator_numbers['PS_mean']
        self.PS_std = all_emulator_numbers['PS_std']
        self.Tb_mean = all_emulator_numbers['Tb_mean']
        self.Tb_std = all_emulator_numbers['Tb_std']
        self.Ts_mean = all_emulator_numbers['Ts_mean']
        self.Ts_std = all_emulator_numbers['Ts_std']
        self.uv_lf_zs = np.array([6,7,8,10])

        self.PS_err = all_emulator_numbers["PS_err"]
        self.Tb_err = all_emulator_numbers["Tb_err"]
        self.Ts_err = all_emulator_numbers['Ts_err']
        self.xHI_err = all_emulator_numbers['xHI_err']
        self.tau_err = all_emulator_numbers['tau_err']
        #log.info('Finished emulator init sucessfully.')

    def predict(self, astro_params):
        r"""
        Call the emulator, evaluate it at the given parameters, restore dimensions.
    
        Parameters
        ----------
        astro_params : p21.AstroParams or np.ndarray or dict
            An array with the nine astro_params input all $\in [0,1]$ OR in the p21.AstroParams input units.
        """
        log.info('Beginning prediction...')
        if isinstance(astro_params, p21.AstroParams) or type(astro_params) == dict:
            log.info('Must convert input params from AstroParams or dict into array.')
            astro_param_keys = ['F_STAR10','ALPHA_STAR','F_ESC10','ALPHA_ESC','M_TURN', 
               't_STAR','L_X','NU_X_THRESH','X_RAY_SPEC_INDEX']
            theta = np.zeros(len(astro_param_keys))
            for i, key in enumerate(astro_param_keys):
                if isinstance(astro_params, p21.AstroParams):
                    theta[i] = astro_params.defining_dict[key]
                else:
                    theta[i] = astro_params[key]
        elif type(astro_params) == np.ndarray:
            theta = astro_params.copy()
        else:
            raise TypeError('theta is in the wrong format. Should be AstroParams object or nine astrophysical parameters in same order as param_keys.')
        if len(theta.shape) == 1:
            theta = theta.reshape([1,-1])
        normed = True
        # Check that theta is normalized, if not, normalise it.
        if isinstance(astro_params, p21.AstroParams) or max(theta.ravel()) > 1 or min(theta.ravel()) < 0:
            log.info('Theta is not normalized.')
            normed = False # to indicate that input params was not normalised
            for i in range(theta.shape[-1]):
                if i == 0 or i == 2 or i == 4 or i == 6:
                    theta[:,i] = np.log10(theta[:,i])
                if i == 7:
                    theta[:,i] /= 1000. #E0 in keV
                theta[:,i] -= self.limits[i,0]
                theta[:,i] /= (self.limits[i,1] - self.limits[i,0])
        log.info('Normed params: %s' % theta)
        log.info('Start emulator prediction...')
        emu_pred = self.model.predict(theta, verbose = False)
        log.info('End emulator prediction.')

        Tb_pred_normed = emu_pred[:,:84] #First 84 numbers of emu prediction are Tb
        xHI_pred = emu_pred[:,84:84*2] # Next 84 numbers are xHI
        Ts_pred_normed = emu_pred[:,2*84:84*3] # Next 84 numbers are Ts
        Ts_undefined_pred = emu_pred[:,84*3] # Right after Ts is the redshift at which Ts becomes undefined
        PS_pred_normed = emu_pred[:,84*3+1:].reshape((theta.shape[0], 60,12)) # The rest is PS

        # Set the xHI < z(Ts undefined) to 0
        xHI_pred_fix = np.zeros(xHI_pred.shape)

        tau = np.zeros(theta.shape[0])
        uvlfs = np.zeros((theta.shape[0], 3, len(self.uv_lf_zs),100))
        for i in range(theta.shape[0]):
            zbin = np.argmin(abs(self.zs - Ts_undefined_pred[i]))
            if xHI_pred[i,zbin] < 1e-1:
                xHI_pred_fix[i,zbin:] = xHI_pred[i,zbin:]
            else:
                xHI_pred_fix[i,:] = xHI_pred[i,:]
            # Use py21cmFAST to analytically calculate UV LF and $\tau_e$
            log.info('Begin tau computation...')
            tau[i] = p21.wrapper.compute_tau(redshifts = self.zs, global_xHI = xHI_pred_fix[i,:])
            log.info('Tau computation completed successfully.')

            # Build astro_params array from theta \in [0,1]
            if (isinstance(astro_params, p21.AstroParams) or type(astro_params) == dict) and not normed:
                uvlfs[i,...] = np.array(p21.wrapper.compute_luminosity_function(redshifts = self.uv_lf_zs, astro_params=astro_params))
            else:
                # Restore dimensions i.e. undo the limits
                log.info('Restore dimensions to theta to calculate UV LF.')
                theta_wdims = np.zeros(theta.shape)
                for j in range(theta.shape[-1]):
                    theta_wdims[:,j] = self.limits[j, 0] + theta[:, j] * (self.limits[j, 1] - self.limits[j, 0])
                    if j == 7:
                        theta_wdims[:,j] = 1000. * theta_wdims[:,j]
                        
                astro_params = {'F_STAR10':theta_wdims[i,0], 'ALPHA_STAR': theta_wdims[i,1], 'F_ESC10': theta_wdims[i,2], 
                                'ALPHA_ESC': theta_wdims[i,3], 'M_TURN': theta_wdims[i,4], 't_STAR': theta_wdims[i,5], 
                                'L_X': theta_wdims[i,6], 'NU_X_THRESH': theta_wdims[i,7], 'X_RAY_SPEC_INDEX':theta_wdims[i,8]}
                log.debug('Restored dimensions are now: ', astro_params)
                log.info('Begin UV LF computation...')
                uvlfs[i,...] = np.array(p21.wrapper.compute_luminosity_function(redshifts = self.uv_lf_zs, astro_params=astro_params))
                if np.sum(np.isnan(uvlfs[i,-1,:,:])) > 300:
                    log.warning('UV LF computation failed: mostly NaNs.')
                else:
                    log.info('UV LF computation completed successfully.')

        # Restore dimensions
        PS_pred = self.PS_mean + self.PS_std * PS_pred_normed # log10(PS[mK^2])
        Ts_pred = self.Ts_mean + self.Ts_std * Ts_pred_normed # log10(Ts[mK])
        Tb_pred = self.Tb_mean + self.Tb_std * Tb_pred_normed # Tb[mK]


        if theta.shape[0] == 1:
            log.info('If only one theta was supplied, flatten all outputs.')
            summaries = {'delta': 10**PS_pred[0,...], 'k': self.ks_cut, 'brightness_temp': Tb_pred[0,...], 
                     'spin_temp': 10**Ts_pred[0,...], 'tau_e': tau[0], 'Muv': uvlfs[0,0,:,:], 'lfunc': uvlfs[0,-1,:,:], 'uv_lfs_redshifts':self.uv_lf_zs,
                     'ps_redshifts':self.zs_cut, 'redshifts': self.zs, 'xHI': xHI_pred_fix[0,...]}
        else:
            summaries = {'delta': 10**PS_pred, 'k': self.ks_cut, 'brightness_temp': Tb_pred, 
                     'spin_temp': 10**Ts_pred, 'tau_e': tau, 'Muv': uvlfs[:,0,:,:], 'lfunc': uvlfs[:,-1,:,:], 'uv_lfs_redshifts':self.uv_lf_zs,
                     'ps_redshifts':self.zs_cut, 'redshifts': self.zs, 'xHI': xHI_pred_fix}
        errors = self.get_errors(summaries, theta)
        # Put the summaries and errors in one single dict
        output = summaries.copy()
        for k in errors.keys():
            output[k] = errors[k]

        if self.io_options is not None and self.io_options['cache_dir'] is not None and len(self.io_options['store']) > 0:
            if isinstance(astro_params, p21.AstroParams) or isinstance(astro_params, dict):
                ap = astro_params.defining_dict
                fname = '_'.join([str(np.round(ap[i], 5)) for i in ap.keys()])
            else:
                fname = '_'.join([str(np.round(astro_params[i], 5)) for i in range(len(theta))])
            to_save = {}
            for i in self.io_options['store']:
                to_save[i] = output[i]
            np.savez(fname, to_save)
            log.info('Successfully wrote data to disk.')

        return output

    def get_errors(self, summaries : dict, theta : np.ndarray = None):
        r"""
        Calculate the emulator error on its outputs.
        Returns the mean error on the test set (i.e. independent of theta).
        
        Parameters
        ----------
        summaries : dict
            Dict containing the emulator predictions, defined in p21cmEMU.predict
        theta : dict
            Dict containing the normalized parameters, also defined in p21cmEMU.predict
        """

        # For now, we return the mean emulator error (obtained from the test set) for each summary.
        # Some errors are fractional => actual error = fractional error * value
        output = {'delta_err': self.PS_err/100. * summaries['delta'], 'brightness_temp_err': self.Tb_err, 'xHI_err': self.xHI_err,
                      'spin_temp_err': self.Ts_err, 'tau_e_err': self.tau_err/100. * summaries['tau_e']}
        return output

