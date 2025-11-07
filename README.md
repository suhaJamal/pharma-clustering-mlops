# DS04-Team-Project


## Team-04 has chosen to analyze the 'Pharmaceutical Drug Spending by Countries' dataset.   


### Our Question

#### What are the clusters among countries in terms of pharmaceutical spending (as a percentage of health spending, percentage of GDP, and per capita spending),  and how do these patterns relate to total spending across different years and inform market entry strategies?


### Our audience

#### Citizens of the respective countries we are analyzing in the dataset. Additionally, this analysis would also be available for regulatory agencies, think-tank organizations and governments to inform public policy decisions.


### Business Motivation

#### Our core motivation is to understand the clusters of countries based on their overall pharmaceutical spending by analyzing it as a percentage of health spending and percentage of GDP and per capita spending. 

#### We want to understand the breakdown and see if our hypothesis that high income countries will be clustered with other high spending countries while middle and low income countries will be clustered with other similar countries. We predict that the United States and Germany will be amoung the higher spenders (if not the highest) on most metrics because they have established and active pharmaceutical industry players who have traditionally spent high sum of money on R&D and new medicinal development.


### If time permits

#### We also want to understand if there are any patterns in the OECD and non-OECD countries and if COVID-19 played a part in increasing pharmaceutical spending in all of the countries of our analysis.


### Methods of analysis

#### We will be using various techniques to exmaine our data. First, we will clean up the data using python scripts to ensure there is no duplication or missing values. Then we will conduct Exploratory Data Analysis through visualization. We will showcase the data using multiple charts and graphs to break it down into bite-sized pieces to help understand our data better. It will also help us see if the trends we are witnessing match our initial hypothesis and help us pivot or refocus our questions if required. Next, we will use a variety of analysis techniques including k-means and hierarchical analysis to understand our data further. Last, we will present our data in a coherent manner so our target audience is able to sift through and understand our results. We will also keep the principle of reproducability at the front so our audience is able to verify the results.


### Risks or Unknowns

#### At this point we see one risk emerging from our analysis, incompleteness. For certain countries, data is missing for several years and is often sporadically missing. Therefore, we will narrow down the dataset to include only those countries whose dataset is not missing any values or missing up to 1 year of data. Our initial analysis shows that most of the countries have most of the years available between 2012 to 2021. Therefore we will focus our analysis on these years. This might result in us missing some trends from earlier years but we thought this is a necessary risk to ensure we don't have missing data bias and we can be confident about our findings. For those countries missing a value we will impute the mean of that country's 10 year history from 2012 to 2021. 


#### There are not any unknowns that we have identified. We will do a comprehensive analysis of our question and if any unknowns later turn into knowns, we will state them in our analysis. 

