import paramater_tuning
import solve_scipy
import read_data
import plot
from common import *

def get_bestParameters():
    # anhui = paramater_tuning.MedicalUpgrade("Anhui", 1000, 2, 12, 19, 34)
    # anhui.get_best_parameters_for_medical_upgrade()
    CI = paramater_tuning.MedicalUpgrade("Channel Islands", 1000, 55, 58, 84, 87)
    CI.get_best_parameters_for_medical_upgrade()


if __name__ == '__main__':
    get_bestParameters()

