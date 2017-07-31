
import numpy as np
import os
import pickle
from astropy.table import Table
from pystan import StanModel

SMALL_VARIANCE = 1e-12
DATA_DIR = os.path.join(os.path.dirname(__file__), "apogee-dr14/")

O = 2 # order of the model
LABEL_NAMES = ["TEFF", "LOGG", "FE_H"]
ERROR_LABEL_NAMES = ["{}_ERR".format(l) for l in LABEL_NAMES]
L = len(LABEL_NAMES)

def get_spectrum_path(star):
    return os.path.join(DATA_DIR, "{0}/cannonStar-r8-l31c.1-{1}.pkl".format(
        star["LOCATION_ID"], star["APOGEE_ID"]))

# Load the training set.
training_set = Table.read(
    os.path.join(DATA_DIR, "apogee-dr14-giants-training-set.fits"))

P = 8575 # number of pixels
S = len(training_set) # number of stars in the training set

training_set_flux = np.ones((S, P), dtype=float)
training_set_ivar = np.zeros((S, P), dtype=float)

for s, star in enumerate(training_set):

    spectrum_path = get_spectrum_path(star)
    assert os.path.exists(spectrum_path), "Cannot find spectrum!"

    try:
        with open(spectrum_path, "rb") as fp:
            flux, ivar = pickle.load(fp)

    except:
        print("Failed to load spectrum in {}".format(spectrum_path))
        continue

    else:
        training_set_flux[s, :] = flux
        training_set_ivar[s, :] = ivar

# Remove spectra with no information.
bad = np.all(training_set_ivar == 0, axis=1)
if sum(bad) > 0:
    print("Removing {} spectra with no information".format(sum(bad)))

    keep = ~bad
    training_set = training_set[keep]
    training_set_flux = training_set_flux[keep]
    training_set_ivar = training_set_ivar[keep]

    S = len(training_set)

# Prepare the payload for the model.
training_set_labels = np.vstack([training_set[ln] for ln in LABEL_NAMES]).T
training_set_errors = np.vstack([training_set[ln] for ln in ERROR_LABEL_NAMES]).T

bad = (training_set_ivar == 0) \
    + ~np.isfinite(training_set_flux) \
    + ~np.isfinite(training_set_ivar) 
training_set_flux[bad] = 1.0
training_set_ivar[bad] = SMALL_VARIANCE



data = {
    "P": P, # number of pixels
    "S": S, # number of training set stars
    "O": O, # order of the model
    "L": L, # number of training set labels

    "y": training_set_flux,
    "y_var": 1.0/training_set_ivar,
    
    "label": training_set_labels,
    "label_err": training_set_errors,
    "f": 1.0
}

init = {
    "eps_variance": np.zeros(P),
    "theta": np.hstack([np.ones((P, 1)), np.zeros((P, 9))]),
    "true_label": training_set_labels,
}

# Load the appropriate model.
model = StanModel(file="cannon-{L}L-O{O}.stan".format(L=L, O=O))

op_params = model.optimizing(data=data, init=init, verbose=True)



