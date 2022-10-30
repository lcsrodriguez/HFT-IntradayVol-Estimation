# HFT data

## Introduction

The HFT datasets have been provided by the free offer of **Kibot**


**Official link**: http://www.kibot.com/free_historical_data.aspx


## Description of the two files

We have downloaded these two files:
1. **`IVE_tickbidask.txt`** -  Tick with bid/ask data (unadjusted)
2. **`IVE_bidask1min.txt`** - Aggregate 1 minute bid/ask data (adjusted)



The first file will be widely used during this project.


> The order of the fields in the tick files (with bid/ask prices) is: **Date,Time,Price,Bid,Ask,Size**. Our bid/ask prices are recorded whenever a trade occurs and they represent the "national best bid and offer" (NBBO) prices across multiple exchanges and ECNs.

> For each trade, current best bid/ask values are recorded together with the transaction price and volume. Trade records are not aggregated and all transactions are included in their consecutive order.

> The order of fields in our regular tick files **(without bid/ask)** is: **Date,Time,Price,Size**.

*Source*: http://www.kibot.com/Support.aspx#tick_data_format


## Miscellaneous

**Remark**: In order to respect the datasets license and the GitHub policies on pushing huge files (>100MB), we do not provide the dataset here. They can freely be downloaded and processed using the above link.
