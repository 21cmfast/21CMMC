import numpy as np
import os
import tensorflow as tf
import py21cmfast as p21
import logging

log = logging.getLogger("21cmFAST")

class p21cmEMU:
    r"""
    This class allows the use to load an emulator and use it to obtain 21cmFAST summaries.
    """
    def __init__(self,
                 emu_path : str,
                 io_options : dict = None,
                     ):
        """
        Parameters
        ----------
        emu_path : str
            Path of the emulator folder location including the folder name e.g. ./folder/21cmEMU
        io_options : dict, optional
            Dict containing 'store' and 'cache_dir' keys with the keys of summaries to store and folder path
            where to store them, respectively. This must be provided only if you want to save the emulator output 
            at each evaluation.
        
        
        """
        log.debug('Init emulator...')
        try:
            emu = tf.keras.models.load_model(emu_path, compile = False)
        except OSError as e:
            raise ValueError('Wrong file path to emulator:', emu_path,' Remember to include the emulator folder name in the path.', e)

        self.model = emu
        self.io_options = io_options

        if len(os.path.dirname(emu_path)) > 1:
            all_emulator_numbers = np.load(os.path.dirname(emu_path) + '/emulator_constants.npz')
        else:
            all_emulator_numbers = np.load('emulator_constants.npz')

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
        
        


    def predict(self, astro_params, cosmo_params = None, user_params = None, flag_options = None):
        r"""
        Call the emulator, evaluate it at the given parameters, restore dimensions.
    
        Parameters
        ----------
        astro_params : p21.AstroParams or np.ndarray or dict
            An array with the nine astro_params input all $\in [0,1]$ OR in the p21.AstroParams input units.
            Dicts and p21.AstroParams are also accepted formats.
            Arrays of only dicts or AstroParams are accepted as well (for batch evaluation).
        """
        self.check_params(cosmo_params, user_params, flag_options)
        astro_params, theta = self.format_theta(astro_params)
   
        emu_pred = self.model.predict(theta, verbose = False)

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

            tau[i] = p21.wrapper.compute_tau(redshifts = self.zs, global_xHI = xHI_pred_fix[i,:], 
                                             cosmo_params = self.cosmo_params, user_params = self.user_params)
            
            uvlfs[i,...] = np.array(p21.wrapper.compute_luminosity_function(redshifts = self.uv_lf_zs, 
                                                                            astro_params=astro_params[i],
                                                                            cosmo_params = self.cosmo_params, 
                                                                            user_params = self.user_params, 
                                                                            flag_options = self.flag_options))
            if np.sum(np.isnan(uvlfs[i,-1,:,:])) > 200:
                log.warning('UV LF computation failed: mostly NaNs.')

        # Restore dimensions
        PS_pred = self.PS_mean + self.PS_std * PS_pred_normed # log10(PS[mK^2])
        Ts_pred = self.Ts_mean + self.Ts_std * Ts_pred_normed # log10(Ts[mK])
        Tb_pred = self.Tb_mean + self.Tb_std * Tb_pred_normed # Tb[mK]


        if theta.shape[0] == 1:

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
    
    def format_theta(self, astro_params):
        astro_param_keys = ['F_STAR10','ALPHA_STAR','F_ESC10','ALPHA_ESC','M_TURN', 
               't_STAR','L_X','NU_X_THRESH','X_RAY_SPEC_INDEX']
        is_astroparams = False
        if isinstance(astro_params, p21.AstroParams):
            is_astroparams = True
            theta = np.array([astro_params.defining_dict[key] for key in astro_param_keys])
        elif isinstance(astro_params, dict):
            theta = np.array([astro_params[key] for key in astro_param_keys])
        elif type(astro_params) == np.ndarray:
            if len(astro_params.shape) > 1 and astro_params.shape[0] > 1:
                #If we supply an array of p21.AstroParams / dict
                theta = np.zeros(astro_params.shape)
                if isinstance(astro_params[0], p21.AstroParams):
                    is_astroparams = True
                    for i in range(astro_params.shape[0]):
                        theta[i,:] = np.array([astro_params[i].defining_dict[key] for key in astro_param_keys])
                elif isinstance(astro_params, dict):
                    for i in range(astro_params.shape[0]):
                        theta = np.array([astro_params[key] for key in astro_param_keys])
                elif type(astro_params[0]) == np.ndarray:
                    theta = astro_params.copy()
                else:
                    raise TypeError('theta is in the wrong format. Should be AstroParams object, dict of astro params or nine astrophysical parameters in same order as astro_param_keys. It can also be an array of either AstroParams objects, dicts, or arrays (not mixed together).')
            else:
                theta = astro_params.copy()
        else:
            raise TypeError('theta is in the wrong format. Should be AstroParams object, dict or nine astrophysical parameters in same order as astro_param_keys. It can also be an array of either AstroParams objects, dicts, or arrays (not mixed together).')
        if len(theta.shape) == 1:
            theta = theta.reshape([1,-1])
        normed = True
        # Check that theta is normalized, if not, normalise it.
        if is_astroparams or max(theta.ravel()) > 1 or min(theta.ravel()) < 0:
            normed = False # to indicate that input params was not normalised         
            theta[:, [0,2,4,6]] = np.log10(theta[:, [0,2,4,6]]) 
            theta[:, 7] /= 1000 
            theta -= self.limits[:, 0] 
            theta /= (self.limits[:, 1] - self.limits[:, 0])
            # Restore dimensions i.e. undo the limits
            all_astro_params = self.undo_normalization(theta)
        
            return all_astro_params, theta
        else:
            if normed == True:
                return self.undo_normalization(theta), theta
            else:
                return np.array([astro_params]), theta
    
    def undo_normalization(self, theta):
        theta_wdims = theta.copy()
        theta_wdims *= (self.limits[:, 1] - self.limits[:, 0])
        theta_wdims += self.limits[:, 0] 
        theta_wdims[:, 7] *= 1000 
        all_astro_params = []
        for i in range(theta.shape[0]):
            all_astro_params.append({'F_STAR10':theta_wdims[i,0], 'ALPHA_STAR': theta_wdims[i,1], 'F_ESC10': theta_wdims[i,2], 
                                     'ALPHA_ESC': theta_wdims[i,3], 'M_TURN': theta_wdims[i,4], 't_STAR': theta_wdims[i,5], 
                                     'L_X': theta_wdims[i,6], 'NU_X_THRESH': theta_wdims[i,7], 'X_RAY_SPEC_INDEX':theta_wdims[i,8]})
        return all_astro_params

    def check_params(self, cosmo_params, user_params, flag_options):
        training_cosmo_params = dict(SIGMA_8=0.82, hlittle=0.6774, OMm=0.3075, 
                                     OMb=0.0486, POWER_INDEX=0.97)
        if cosmo_params is not None:
            if isinstance(cosmo_params, p21.CosmoParams):
                self.cosmo_params = cosmo_params
            else:
                self.cosmo_params = p21.CosmoParams(cosmo_params)
        
            ## Check that given cosmo params match emulator training data cosmo params
            ## if they do not, raise error and exit
            for key in emu_cosmo_params.keys():
                if self.cosmo_params.defining_dict[key] != training_cosmo_params[key]:
                    raise ValueError('Input cosmo_params do not match the emulator cosmo_params. The emulator can only be used with a single set of cosmo params:', training_cosmo_params)
        else:
            self.cosmo_params = p21.CosmoParams(training_cosmo_params)
        
        training_flag_options = {"USE_HALO_FIELD": False, "USE_MINI_HALOS": False,
                                     "USE_MASS_DEPENDENT_ZETA": True, "SUBCELL_RSD": True,
                                     "INHOMO_RECO": True, "USE_TS_FLUCT": True,
                                     "M_MIN_in_Mass": False,"PHOTON_CONS": True,
                                     "FIX_VCB_AVG": False, "EVOLVING_R_BUBBLE_MAX": False}
        if flag_options is not None:
            if isinstance(flag_options, p21.FlagOptions):
                self.flag_options = flag_options
            else:
                self.flag_options = p21.FlagOptions(flag_options)
        
            ## Check that given flag options match emulator training data flag options
            ## if they do not, raise error and exit
            for key in training_flag_options.keys():
                if self.flag_options.defining_dict[key] != training_flag_options[key]:
                    raise ValueError('Input flag options do not match the emulator flag options. The emulator can only be used with a single set of flag options:', training_flag_options)
        else:
            self.flag_options = training_flag_options
        
        training_user_params = {"BOX_LEN": 250, "DIM": 512,
                                    "HII_DIM": 128,"USE_FFTW_WISDOM": True,
                                    "HMF": 1,"USE_RELATIVE_VELOCITIES": False,
                                    "POWER_SPECTRUM": 0,"N_THREADS": 1,
                                    "PERTURB_ON_HIGH_RES": False,"NO_RNG": False,
                                    "USE_INTERPOLATION_TABLES": True,"FAST_FCOLL_TABLES": False,
                                    "USE_2LPT": True,"MINIMIZE_MEMORY": False}
        if user_params is not None:
            if isinstance(user_params, p21.UserParams):
                self.user_params = user_params
            else:
                self.user_params = p21.UserParams(user_params)
        
            ## Check that given flag options match emulator training data flag options
            ## if they do not, raise error and exit
            for key in training_user_params.keys():
                if self.user_params.defining_dict[key] != training_user_params[key]:
                    raise ValueError('Input user params do not match the emulator user params. The emulator can only be used with a single set of user params:', training_user_params)
        else:
            self.user_params = training_user_params

