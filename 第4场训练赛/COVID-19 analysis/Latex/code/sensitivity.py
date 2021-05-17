import read_data
import paramater_tuning
import plot
from common import *


def compare_hStart(province, change=5):
    # DATA_JRD = np.array(read_data.read_cleaned_data(province)[
    #                         ["confirmedMA", "recoveriesMA", "deathsMA"]][:DAYS].fillna(0))
    seir_solutions = []
    bestParameters = bestParametersDict[province]
    hStarts = (bestParameters["hStart"]-change, bestParameters["hStart"], bestParameters["hStart"]+5)
    for hStart in hStarts:
        bestParameters["hStart"] = hStart
        seir_solution = paramater_tuning.get_medical_upgrade_solution(bestParameters)
        seir_solution = np.diff(seir_solution, axis=0)
        seir_solutions.append(seir_solution)
    plot.compare_results(seir_solutions, "hStart", "隔离措施开始时间", hStarts, province=province)


if __name__ == '__main__':
    compare_hStart("Channel Islands")
    compare_hStart("Anhui")