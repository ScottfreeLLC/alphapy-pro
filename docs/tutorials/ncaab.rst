NCAA Basketball Tutorial (Deprecated)
====================================

.. warning::
   This tutorial is deprecated as SportFlow is no longer available in AlphaPy Pro.
   
   For sports prediction using AlphaPy Pro, see the :doc:`../user_guide/sport_flow`
   migration guide.

Historical Context
------------------

This tutorial previously demonstrated using SportFlow to predict NCAA Men's Basketball
game outcomes, specifically whether a team would cover the spread.

The original workflow involved:

1. **Data Collection** - Game results, lines, and over/under totals
2. **Feature Engineering** - Time series features like streaks and runs
3. **Model Training** - Binary classification for spread coverage
4. **Evaluation** - ROC curves and prediction accuracy

Migration Path
--------------

To recreate similar functionality using current AlphaPy Pro:

**1. Data Preparation:**

Create a CSV file with your basketball data:

.. code-block:: csv

    date,home_team,away_team,home_score,away_score,line,covers_spread
    2024-01-15,Duke,UNC,78,72,-3,1
    2024-01-16,Kansas,Kentucky,85,81,2,1

**2. Feature Engineering:**

Use pandas to create sports-specific features:

.. code-block:: python

    import pandas as pd
    
    def create_basketball_features(df):
        # Calculate win streaks
        df['home_win_streak'] = df.groupby('home_team')['home_wins'].rolling(5).sum()
        
        # Average scoring
        df['home_avg_score'] = df.groupby('home_team')['home_score'].rolling(10).mean()
        
        # Rest days
        df['rest_days'] = df.groupby('home_team')['date'].diff().dt.days
        
        return df

**3. Model Configuration:**

Create a ``config/model.yml`` for classification:

.. code-block:: yaml

    model:
        algorithms: ['CATB', 'LGB', 'XGB']
        type: classification
        target: covers_spread
        scoring_function: roc_auc

**4. Run AlphaPy:**

.. code-block:: bash

    cd your_basketball_project
    alphapy

Alternative Solutions
---------------------

For comprehensive sports analytics, consider:

**Specialized Libraries:**
- ``sportsreference`` for historical data
- ``nba_api`` for NBA-specific data
- Custom scrapers for college sports

**Commercial Platforms:**
- Sports betting analytics platforms
- Fantasy sports modeling tools
- Professional sports data providers

**Academic Resources:**
- Sports analytics research papers
- University sports analytics programs
- Open source sports modeling projects

The principles from this tutorial—feature engineering, time series analysis,
and binary classification—remain valid and can be applied using the core
AlphaPy Pro framework with custom data preprocessing.