from common import *
import plot


def read_data(province='Hubei'):
    confirmed_df = pd.read_csv(
        '../input/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    confirmed_df = confirmed_df[confirmed_df['Province/State'] == province]
    deaths_df = pd.read_csv(
        '../input/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    deaths_df = deaths_df[deaths_df['Province/State'] == province]
    recoveries_df = pd.read_csv(
        '../input/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    recoveries_df = recoveries_df[recoveries_df['Province/State'] == province]

    provinceDF = pd.DataFrame(
        index=confirmed_df.columns,
        columns=["confirmed", "deaths", "recoveries"])

    provinceDF["confirmed"] = confirmed_df.T
    provinceDF["deaths"] = deaths_df.T
    provinceDF["recoveries"] = recoveries_df.T

    provinceDF = provinceDF[4:]
    #provinceDF.index = pd.date_range("20200122", "20200720")

    provinceDF["confirmedMA"] = provinceDF["confirmed"].rolling(window=WINDOW, center=True).mean()
    provinceDF["deathsMA"] = provinceDF["deaths"].rolling(window=WINDOW, center=True).mean()
    provinceDF["recoveriesMA"] = provinceDF["recoveries"].rolling(window=WINDOW, center=True).mean()

    provinceDF["confirmedDaily"] = provinceDF["confirmed"].diff()
    provinceDF["deathsDaily"] = provinceDF["deaths"].diff()
    provinceDF["recoveriesDaily"] = provinceDF["recoveries"].diff()

    provinceDF["confirmedDailyMA"] = provinceDF["confirmedDaily"].rolling(window=WINDOW, center=True).mean()
    provinceDF["deathsDailyMA"] = provinceDF["deathsDaily"].rolling(window=WINDOW, center=True).mean()
    provinceDF["recoveriesDailyMA"] = provinceDF["recoveriesDaily"].rolling(window=WINDOW, center=True).mean()
    plot.plot_provinceDF(provinceDF, province)
    return provinceDF.plot


def read_cleaned_data(province='Hubei'):
    try:
        return pd.read_excel("../cache/{}.xlsx".format(province))
    except FileNotFoundError:
        print("Preprocessing data..")
    df = pd.read_csv(
        '../input/covid_19_clean_complete.csv')
    provinceDF = df[df['Province/State'] == province]
    provinceDF.index = pd.date_range("20200122", "20200727", freq='1D')

    provinceDF["confirmedMA"] = provinceDF["confirmed"].rolling(window=WINDOW, center=True).mean()
    provinceDF["deathsMA"] = provinceDF["deaths"].rolling(window=WINDOW, center=True).mean()
    provinceDF["recoveriesMA"] = provinceDF["recoveries"].rolling(window=WINDOW, center=True).mean()

    provinceDF["confirmedDaily"] = provinceDF["confirmed"].diff()
    provinceDF["deathsDaily"] = provinceDF["deaths"].diff()
    provinceDF["recoveriesDaily"] = provinceDF["recoveries"].diff()

    provinceDF["confirmedDailyMA"] = provinceDF["confirmedDaily"].rolling(window=WINDOW, center=True).mean()
    provinceDF["deathsDailyMA"] = provinceDF["deathsDaily"].rolling(window=WINDOW, center=True).mean()
    provinceDF["recoveriesDailyMA"] = provinceDF["recoveriesDaily"].rolling(window=WINDOW, center=True).mean()
    plot.plot_provinceDF(provinceDF, province)
    provinceDF.to_excel("../cache/{}.xlsx".format(province))
    return provinceDF.plot


if __name__ == '__main__':
    read_cleaned_data(province="Channel Islands")
    read_cleaned_data(province="Anhui")

