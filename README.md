# Exploratory Data Analysis
---

## Classification Problem

```python
import pandas as pd
from fast_ml.tools import EDA

df = pd.read_csv('titanic.csv')

eda_report = EDA.generate_report(df, target='Survived', model_type='clf')
eda_report.report_title_ = 'EDA Report for Titanic Dataset (Classification)'
eda_report.report_user_ = 'Samarth Agrawal (using Fast ML)'
eda_report.show()
```
---

## Regression Problem

```python
import pandas as pd
from fast_ml.tools import EDA

df = pd.read_csv('house_prices.csv')

eda_report = EDA.generate_report(df, target='SalePrice', model_type='reg')
eda_report.report_title_ = 'EDA Report for House Price Dataset (Regression)'
eda_report.report_user_ = 'Samarth Agrawal (using Fast ML)'
eda_report.show()
```
