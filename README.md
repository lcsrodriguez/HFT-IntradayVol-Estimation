# Intraday volatility estimation from high frequency data

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lcsrodriguez/HFT-IntradayVol-Estimation/HEAD?urlpath=https%3A%2F%2Fgithub.com%2Flcsrodriguez%2FHFT-IntradayVol-Estimation%2Fblob%2Fmain%2Fmain.ipynb)

## Overview

The aim is to estimate the daily volatility and assess the effect of micro-structure noise in high-frequency data.

The main objectives of this project are:

1. Estimate and plot the values of the **estimated realized volatility** when using observation frequencies ranging from 30 seconds to 15 minutes.
2. Compare these estimations with the long range estimation of the volatility (based on 1 month of daily data)
3. Provide some estimation of the micro-structure noise size (by using the autocorrelation between the returns at different scales)
4. Plot the evolution of the estimated daily volatility of the IVE over the last year.


*The academic report highlighting the research tracks, solutions and main results is available [here](docs/report.pdf).*


Please first install all the requirements in order to run the Jupyter Notebook
```shell
pip3 install -r requirements.txt
jupyter-notebook main.ipynb
```

## Dataset source


The dataset is extracted from the free data samples offered by **Kibot**.

To update the used dataset to the most updated information, please execute the following command:


```shell
cd data/
URL=http://api.kibot.com/?action=history&symbol=IVE&interval=tickbidask&bp=1&user=guest
wget $URL
```


**Remarks**
- In order to compare the realized volatility over several time frequencies, we have to slice the observation samples with respect to the time column with the given time step, because we cannot download data for each asked frequency (30 sec, 1 min, ...).
- All the relevant images (charts, tables) are available in the folders `out/` or `img/`.

## [UPDATE]

The presented approach uses an average resampling, by implementing the following procedure:

```python
# df: price dataframe/series (pandas)
freq: str = "1D" # resampling frequency
df.resample(freq).mean() # the .mean() represents the aggregation method used. 
```

However, since we assume that the price of the asset at time $t$ is given by the value of the closest (in time) past transaction, we are strongly encourage to take the value of the left of each sampled interval, instead of the mean (the mean increases the autocovariance since the same element is captured twice per computation, and we cannot recover the negative autocorrelation effect when we study the $\vartheta$ size).

This means that the research work presented in the report is taking into account a pre-averaging method, in order to capture and reduce the micro-structure noise $\vartheta$.

## License

**Lucas RODRIGUEZ**

*Academic work - October/November/December 2022 & January 2023*


