import pyflex


def config_repo(mode=None, period=None, component=None, depth_mode=None):

    mode = mode.lower()
    if mode in "surface_waves":
        if component.lower() in ["z", "r"]:
            config = config_sw_zr(period)
        elif component.lower() in ["t"]:
            config = config_sw_t(period)
        else:
            raise ValueError("Component not recognised: %s" % component)
    elif mode in "body_waves":
        if component.lower() in ["z", "r"]:
            config = config_bw_zr(period)
        elif component.lower() in ["t"]:
            config = config_bw_t(period)
        else:
            raise ValueError("Component not recognised: %s" % component)
    else:
        raise ValueError("mode not recognised: %s" % mode)
    return config


def config_bw_zr(period):
    return pyflex.Config(
            min_period=float(period[0]), max_period=float(period[1]),
            stalta_waterlevel=0.10, tshift_acceptance_level=8.0,
            dlna_acceptance_level=0.50, cc_acceptance_level=0.90, s2n_limit=3.0,
            min_surface_wave_velocity=3.20, max_surface_wave_velocity=4.10,
            check_global_data_quality=True,
            c_0=0.7, c_1=2.0, c_2=0.0, c_3a=1.0, c_3b=2.0, c_4a=3.0, c_4b=10.0,
            window_signal_to_noise_type="amplitude",
            selection_mode="body_waves")


def config_bw_t(period):
    return pyflex.Config(
            min_period=float(period[0]), max_period=float(period[1]),
            stalta_waterlevel=0.10, tshift_acceptance_level=8.0,
            dlna_acceptance_level=0.60, cc_acceptance_level=0.90, s2n_limit=3.0,
            min_surface_wave_velocity=3.80, max_surface_wave_velocity=4.80,
            check_global_data_quality=True,
            c_0=0.7, c_1=2.0, c_2=0.0, c_3a=1.0, c_3b=2.0, c_4a=3.0, c_4b=10.0,
            window_signal_to_noise_type="amplitude",
            selection_mode="body_waves")


def config_sw_zr(period):
    return pyflex.Config(
            min_period=float(period[0]), max_period=float(period[1]),
            stalta_waterlevel=0.10, tshift_acceptance_level=15.0,
            dlna_acceptance_level=0.50, cc_acceptance_level=0.90, s2n_limit=3.5,
            min_surface_wave_velocity=3.2, max_surface_wave_velocity=4.2,
            check_global_data_quality=True,
            c_0=0.7, c_1=3.0, c_2=0.0, c_3a=1.0, c_3b=2.0, c_4a=3.0, c_4b=10.0,
            window_signal_to_noise_type="amplitude",
            selection_mode="surface_waves")


def config_sw_t(period):
    return pyflex.Config(
            min_period=float(period[0]), max_period=float(period[1]),
            stalta_waterlevel=0.10, tshift_acceptance_level=15.0,
            dlna_acceptance_level=0.50, cc_acceptance_level=0.90, s2n_limit=3.5,
            min_surface_wave_velocity=3.8, max_surface_wave_velocity=5.0,
            check_global_data_quality=True,
            c_0=0.7, c_1=3.0, c_2=0.0, c_3a=1.0, c_3b=2.0, c_4a=3.0, c_4b=10.0,
            window_signal_to_noise_type="amplitude",
            selection_mode="surface_waves")
