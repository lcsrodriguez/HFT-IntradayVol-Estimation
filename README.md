# Intraday volatility estimation from high frequency data

## Overview

The aim is to estimate the daily volatility and assess the effect of micro-structure noise in high-frequency data.

The main objectives of this project are:

1. Estimate and plot the values of the **estimated realized volatility** when using observation frequencies ranging from 30 seconds to 15 minutes.
2. Compare these estimations with the long range estimation of the volatility (based on 1 month of daily data)
3. Provide some estimation of the micro-structure noise size (by using the autocorrelation between the returns at different scales)
4. Plot the evolution of the estimated daily volatility of the IVE over the last year.


*The academic report highlighting the research tracks, solutions and main results is available [here](docs/report.pdf).*

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


## License

**Lucas RODRIGUEZ**

*Academic work - October/November/December 2022 & January 2023*


