SportFlow (Deprecated)
=====================

.. warning::
   SportFlow is no longer available in AlphaPy Pro. This functionality has been
   removed from the current version of the framework.

Migration Guide
---------------

If you were using SportFlow for sports prediction, consider these alternatives:

**Option 1: Use Core AlphaPy Pro Pipeline**

You can still build sports prediction models using the core AlphaPy Pro pipeline:

1. **Prepare your sports data** in CSV format with features and target variables
2. **Create a standard project structure**::

    sports_project/
    ├── config/
    │   └── model.yml
    └── data/
        ├── train.csv
        └── test.csv

3. **Configure your model** for classification or regression
4. **Run the pipeline** using ``alphapy``

**Option 2: Create a Custom Domain Pipeline**

Following the MarketFlow pattern, you can create a custom sports pipeline:

1. **Study the MarketFlow implementation** in ``alphapy/mflow_main.py``
2. **Create your own sports data processor** that:
   - Fetches sports data from APIs
   - Engineers sport-specific features
   - Prepares data for the core ML pipeline
3. **Register your pipeline** as a new entry point

Example Sports Model Configuration
----------------------------------

Here's how you might configure a sports prediction model using core AlphaPy Pro:

.. code-block:: yaml

    project:
        directory: .
        file_extension: csv
        submission_file: 'predictions'

    model:
        algorithms: ['CATB', 'LGB', 'XGB', 'RF']
        type: classification
        target: 'team_wins'           # Target: 1 if team wins, 0 if loses
        cv_folds: 5
        scoring_function: roc_auc

    data:
        features: '*'
        drop: ['game_id', 'date', 'team_name', 'opponent']
        split: 0.2

    features:
        clustering:
            option: True
        encoding:
            type: target               # Good for categorical team/venue data
        interactions:
            option: True
            poly_degree: 2
        scaling:
            option: True
            type: standard

Sample Sports Features
----------------------

Common features for sports prediction models:

**Team Performance Metrics:**
- Win/loss streaks
- Points scored vs. points allowed averages
- Home/away performance splits
- Rest days between games

**Matchup Features:**
- Head-to-head historical records
- Style matchups (offense vs. defense rankings)
- Venue factors
- Weather conditions (for outdoor sports)

**Advanced Metrics:**
- Team efficiency ratings
- Strength of schedule
- Recent form indicators
- Injury reports impact

**Example Feature Engineering:**

.. code-block:: python

    # Example of creating sports features manually
    import pandas as pd
    
    def create_sports_features(df):
        # Win streak feature
        df['win_streak'] = df.groupby('team')['win'].transform(
            lambda x: x.rolling(window=10).sum()
        )
        
        # Average points in last N games
        df['avg_points_last_5'] = df.groupby('team')['points'].transform(
            lambda x: x.rolling(window=5).mean()
        )
        
        # Home field advantage
        df['home_advantage'] = df['venue'] == 'home'
        
        return df

Resources for Sports Analytics
------------------------------

If you're interested in sports analytics, consider these resources:

**Data Sources:**
- Sports Reference APIs
- ESPN APIs
- Custom web scraping solutions

**Python Libraries:**
- ``sportsreference`` - College and professional sports data
- ``nba_api`` - NBA data
- ``nfl_data_py`` - NFL data
- ``hockey_scraper`` - NHL data

**Alternative Frameworks:**
- Custom pandas-based pipelines
- Specialized sports analytics platforms
- R-based sports modeling packages

For questions about migrating SportFlow functionality or implementing sports
prediction models, please refer to the core AlphaPy Pro documentation or
create custom domain pipelines following the MarketFlow example.