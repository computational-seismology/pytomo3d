import pyadjoint


def config_setup(mode=None, period=None):

    mode = mode.lower()

    if mode in "cc_traveltime_misfit":
        config = config_cc(period)
    elif mode in "waveform_misfit":
        config = config_wf(period)
    elif mode in "multitaper_misfit":
        config = config_mt(period)
    else:    
        config = config_mt(period)

    return config


def config_cc(period):
    return pyadjoint.Config(
            min_period=float(period[0]), 
            max_period=float(period[1]),
            ipower_costaper=10,
            taper_percentage=0.15,
            taper_type='hann',
            use_cc_error=True)

def config_wf(period):
    return pyadjoint.Config(
            min_period=float(period[0]), 
            max_period=float(period[1]),
            taper_percentage=0.15,
            taper_type='hann',
            use_cc_error=False)

def config_mt(period):
    return pyadjoint.Config(
            min_period=float(period[0]), 
            max_period=float(period[1]),
            lnpt = 15,
            transfunc_waterlevel=1.0E-10,
            ipower_costaper=10,
            min_cycle_in_window=3,
            taper_percentage=0.3,
            mt_nw=4.0,
            num_taper=5,
            phase_step=1.5,
            dt_fac=2.0,
            err_fac=2.5,
            dt_max_scale=3.5,
            measure_type='dt',
            taper_type='hann',
            use_cc_error=True,
            use_mt_error=False)
