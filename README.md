### Baseline Agent for Auto Trading

1. This is a very simple trading agent that uses statistics from training data to decide if the agent should buy, sell, or hold stock.
2. Agent take the training data, calculate the statistics(mean) from this dataset.
3. The statistic is trained online, not offline (no pre-training), before the testing data is ingested
4. Once the statistics is calculated, the results is used to let agent decide .
