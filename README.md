## Q-factor measurement tool and calculator

To install dependencies run `pip install keyoscacquire scipy`

The module can be used also without using keyoscacquire, but `read_data()` must
be changed accordingly, as well as the part of the programme that acquires
traces.

Can be used to only calculate the Q-factor with a known laser frequency sweep
per second calibration, or fit a trace with known sidebands to calculate this
calibration.

The programme can acquire traces as necessary if the data has not already been
taken, run with option `-h` to learn about the syntax.

The module variable `_oscilloscope_visa_address` can be set to the VISA address
of the oscilloscope in use. (To find the VISA address run `list_visa_devices`
from the command line.) Or leave empty to use default VISA address for
keyoscacquire(which can be set by changing the file whose path is given by
running `path_of_config` from the command line).


Run `qfactor.py -h` to learn more about the syntax of use.
