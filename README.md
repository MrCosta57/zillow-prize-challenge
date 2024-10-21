# Predictive Real Estate: Advancing the Zestimate Model

## Description

This project addresses the <a href="https://www.kaggle.com/c/zillow-prize-1/overview">Zillow Prize challenge</a>, which aims to enhance the accuracy of Zillow's Zestimate home valuation model. The Zestimate has revolutionized the U.S. real estate industry by providing consumers with access to estimated home values based on extensive data analysis. Leveraging 7.5 million statistical and machine learning models, Zillow continuously refines its estimates, achieving a median margin of error reduction from 14% to 5%.

The primary objective of this project is to develop a predictive model that accurately estimates house prices. Significant emphasis was placed on feature engineering, data cleaning, and imputation of missing values to ensure data integrity and model performance. The final model demonstrates strong predictive capabilities and high interpretability, facilitated by the choice of machine learning trees.

The insights generated from this work aim to contribute to the ongoing efforts to improve the Zestimate, impacting the home valuations of 110 million properties across the United States.


## Data

Datasets are provided by Zillow for the <a href="https://www.kaggle.com/c/zillow-prize-1/data
">Zillow Prize competition</a>. They consists of various files containing property features, transaction details, and a data dictionary for understanding the available features. Below are the key files used in this analysis:

- **zillow_data_dictionary.xlsx**: This file provides a detailed description of all the fields and features available in the dataset, essential for understanding and interpreting the data correctly.

- **properties_2016.csv**: Contains the property features for 2016, including details such as home size, location, and architectural attributes. This dataset forms the basis for feature extraction and model training. Some properties from 2017 (not used) only have their parcelid without full data, and will be updated when the properties_2017.csv file becomes available.

- **train_2016.csv**: The training dataset, which includes home transaction data from January 1, 2016, to December 31, 2016. This file is used to build and validate the predictive model for home prices.