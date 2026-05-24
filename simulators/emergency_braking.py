# Noise model — AV-specific distributions
# Braking efficiency: TruncatedNormal(mean=0.97, std=0.02, low=0.85, high=1.0)


from scipy.stats import truncnorm

BRAKING_EFFICIENCY_MEAN  = 0.97
BRAKING_EFFICIENCY_STD   = 0.02
BRAKING_EFFICIENCY_LOW   = 0.85
BRAKING_EFFICIENCY_HIGH  = 1.0

DELAY_NOISE_LOG_SIGMA    = 0.1   # log-normal shape parameter
HARDWARE_MIN_DELAY       = 0.05  # minimum physically possible delay (seconds)
