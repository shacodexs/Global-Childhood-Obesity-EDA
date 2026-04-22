
# EDA: Global Childhood Obesity (Age: 5-19) BMI Dataset (NCD_BMI_PLUS2C)



# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# 2. LOADING DATASET

df = pd.read_csv(r"C:\Users\monal\Downloads\body-mass-index\data\NCD_BMI_PLUS2C.csv")


# 3. RENAMING COLUMNS

df.rename(columns={
    'Id'                                  : 'Record_ID',
    'IndicatorCode'                       : 'Indicator_Code',
    'SpatialDimension'                    : 'Location_Type',          # COUNTRY / REGION / GLOBAL
    'SpatialDimensionValueCode'           : 'Country_Code',           # ISO-3 code e.g. IND, USA
    'ParentLocationCode'                  : 'WHO_Region_Code',        # e.g. SEAR, AMR
    'ParentLocation'                      : 'WHO_Region',             # e.g. South-East Asia
    'TimeDimension'                       : 'Time_Dimension_Type',    # always YEAR here
    'TimeDim'                             : 'Year',                   # numeric year
    'DisaggregatingDimension1'            : 'Sex_Dimension',          # label: SEX
    'DisaggregatingDimension1ValueCode'   : 'Sex',                    # SEX_BTSX / SEX_MLE / SEX_FMLE
    'DisaggregatingDimension2'            : 'Age_Dimension',          # label: AGEGROUP
    'DisaggregatingDimension2ValueCode'   : 'Age_Group',              # e.g. AGEGROUP_YEARS05-09
    'DisaggregatingDimension3'            : 'Extra_Dimension',        # empty in this dataset
    'DisaggregatingDimension3ValueCode'   : 'Extra_Dimension_Code',   # empty in this dataset
    'DataSourceDimension'                 : 'Data_Source_Dimension',  # empty in this dataset
    'DataSourceDimensionValueCode'        : 'Data_Source_Code',       # empty in this dataset
    'Value'                               : 'BMI_Value_With_CI',      # e.g. "16.6 [14.1-19.3]"
    'NumericValue'                        : 'BMI_Percent',            # numeric BMI obesity prevalence %
    'Low'                                 : 'CI_Lower_Bound',         # 95% confidence interval lower
    'High'                                : 'CI_Upper_Bound',         # 95% confidence interval upper
    'Comments'                            : 'Comments',
    'Date'                                : 'Record_Date',            # when record was published
    'TimeDimensionValue'                  : 'Year_Label',             # same as Year, string form
    'TimeDimensionBegin'                  : 'Year_Start_Date',        # e.g. 2014-01-01
    'TimeDimensionEnd'                    : 'Year_End_Date',          # e.g. 2014-12-31
}, inplace=True)

print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"Shape      : {df.shape}")
print(f"\nRenamed Columns:\n{list(df.columns)}")
print(f"\nFirst 5 Rows:\n{df.head()}")


# 4. DATA CLEANING

# Droping columns that are entirely empty or irrelevant
df.drop(columns=[
    'Extra_Dimension',
    'Extra_Dimension_Code',
    'Data_Source_Dimension',
    'Data_Source_Code',
    'Comments'
], inplace=True)

# Filling missing categorical values
df['WHO_Region']      = df['WHO_Region'].fillna("Unknown")
df['WHO_Region_Code'] = df['WHO_Region_Code'].fillna("Unknown")

# Ensuring numeric types
df['BMI_Percent']    = pd.to_numeric(df['BMI_Percent'],    errors='coerce')
df['CI_Lower_Bound'] = pd.to_numeric(df['CI_Lower_Bound'], errors='coerce')
df['CI_Upper_Bound'] = pd.to_numeric(df['CI_Upper_Bound'], errors='coerce')
df['Year']           = pd.to_numeric(df['Year'],           errors='coerce')

# Decoding Sex codes
sex_map = {'SEX_BTSX': 'Both Sexes', 'SEX_MLE': 'Male', 'SEX_FMLE': 'Female'}
df['Sex'] = df['Sex'].map(sex_map)

# Decode Age Group codes
age_map = {
    'AGEGROUP_YEARS05-09': '5-9 years',
    'AGEGROUP_YEARS10-19': '10-19 years',
    'AGEGROUP_YEARS05-19': '5-19 years'
}
df['Age_Group'] = df['Age_Group'].map(age_map)


# 5. FILTERING ONLY COUNTRY-LEVEL DATA

df_country = df[df['Location_Type'] == 'COUNTRY'].copy()
print(f"\nCountry-level rows : {df_country.shape[0]}")
print(f"Columns kept       : {df_country.shape[1]}")


# 6. MISSING VALUES

print("\n" + "=" * 55)
print("MISSING VALUES (Country-level data)")
print("=" * 55)
missing     = df_country.isnull().sum()
missing_pct = (missing / len(df_country) * 100).round(2)
missing_df  = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])


# 7. SUMMARY STATISTICS

print("\n" + "=" * 55)
print("SUMMARY STATISTICS — BMI_Percent (Obesity Prevalence %)")
print("=" * 55)
print(df_country['BMI_Percent'].describe().round(3))

# 8. UNIQUE VALUES & COVERAGE

print("\n" + "=" * 55)
print("DATASET COVERAGE")
print("=" * 55)
print(f"Unique Countries : {df_country['Country_Code'].nunique()}")
print(f"Year Range       : {int(df_country['Year'].min())} - {int(df_country['Year'].max())}")
print(f"Sex Groups       : {df_country['Sex'].dropna().unique().tolist()}")
print(f"Age Groups       : {df_country['Age_Group'].dropna().unique().tolist()}")
print(f"WHO Regions      : {df_country['WHO_Region'].unique().tolist()}")


# 9. GROUPING & AGGREGATIONS


# 9a. Overall BMI trend by year (Both Sexes)
yearly_avg = (
    df_country[df_country['Sex'] == 'Both Sexes']
    .groupby('Year')['BMI_Percent']
    .mean()
    .reset_index()
    .rename(columns={'BMI_Percent': 'Avg_BMI_Percent'})
)

# 9b. Sex-wise trend (Age 5-19)
sex_yearly = (
    df_country[df_country['Age_Group'] == '5-19 years']
    .groupby(['Year', 'Sex'])['BMI_Percent']
    .mean()
    .reset_index()
    .rename(columns={'BMI_Percent': 'Avg_BMI_Percent'})
)

# 9c. Country averages (Both Sexes) -> top & bottom 10
country_avg = (
    df_country[df_country['Sex'] == 'Both Sexes']
    .groupby(['Country_Code', 'WHO_Region'])['BMI_Percent']
    .mean()
    .reset_index()
    .rename(columns={'BMI_Percent': 'Avg_BMI_Percent', 'Country_Code': 'Country'})
    .sort_values('Avg_BMI_Percent', ascending=False)
)
top10    = country_avg.head(10)
bottom10 = country_avg.tail(10).sort_values('Avg_BMI_Percent')

# 9d. Regional trend by year
regional_yearly = (
    df_country[df_country['Sex'] == 'Both Sexes']
    .groupby(['Year', 'WHO_Region'])['BMI_Percent']
    .mean()
    .reset_index()
    .rename(columns={'BMI_Percent': 'Avg_BMI_Percent'})
)

# 9e. Age group trend by year
age_trend = (
    df_country[df_country['Sex'] == 'Both Sexes']
    .groupby(['Year', 'Age_Group'])['BMI_Percent']
    .mean()
    .reset_index()
    .rename(columns={'BMI_Percent': 'Avg_BMI_Percent'})
)


# 10. TOP/BOTTOM TABLES

print("\n" + "=" * 55)
print("TOP 10 COUNTRIES — Highest Average Obesity Prevalence (%)")
print("=" * 55)
print(top10[['Country', 'WHO_Region', 'Avg_BMI_Percent']].to_string(index=False))

print("\n" + "=" * 55)
print("BOTTOM 10 COUNTRIES — Lowest Average Obesity Prevalence (%)")
print("=" * 55)
print(bottom10[['Country', 'WHO_Region', 'Avg_BMI_Percent']].to_string(index=False))


# 11. CORRELATION MATRIX

print("\n" + "=" * 55)
print("CORRELATION MATRIX")
print("=" * 55)
corr_cols   = ['BMI_Percent', 'CI_Lower_Bound', 'CI_Upper_Bound', 'Year']
correlation = df_country[corr_cols].corr().round(3)
print(correlation)


# 12. OUTLIER ANALYSIS (IQR)

'''Q1       = df_country['BMI_Percent'].quantile(0.25)
Q3       = df_country['BMI_Percent'].quantile(0.75)
IQR      = Q3 - Q1
outliers = df_country[
    (df_country['BMI_Percent'] < Q1 - 1.5 * IQR) |
    (df_country['BMI_Percent'] > Q3 + 1.5 * IQR)
]
print("\n" + "=" * 55)
print("OUTLIER ANALYSIS (IQR Method) — BMI_Percent")
print("=" * 55)
print(f"Q1 (25th percentile)     : {Q1:.3f}")
print(f"Q3 (75th percentile)     : {Q3:.3f}")
print(f"IQR                      : {IQR:.3f}")
print(f"Lower fence (Q1-1.5xIQR) : {Q1 - 1.5*IQR:.3f}")
print(f"Upper fence (Q3+1.5xIQR) : {Q3 + 1.5*IQR:.3f}")
print(f"Outlier rows             : {len(outliers)} ({len(outliers)/len(df_country)*100:.1f}%)")'''


# 13. VISUALIZATIONS

COLORS = {
    'blue':   '#378add',
    'green':  '#1d9e75',
    'coral':  '#d85a30',
    'amber':  '#ba7517',
    'purple': '#7f77dd',
    'pink':   '#d4537e',
}
REGION_COLORS = ['#378add', '#1d9e75', '#d85a30', '#ba7517', '#a32d2d', '#7f77dd']

plt.rcParams.update({
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
    'font.size':         11,
})

# Plot 1: Overall BMI Trend
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(yearly_avg['Year'], yearly_avg['Avg_BMI_Percent'],
        color=COLORS['blue'], linewidth=2.5, marker='o', markersize=4)
ax.fill_between(yearly_avg['Year'], yearly_avg['Avg_BMI_Percent'],
                alpha=0.12, color=COLORS['blue'])
ax.set_title("Global Childhood Obesity Trend (Both Sexes, 1990-2022)", fontsize=13, pad=12)
ax.set_xlabel("Year")
ax.set_ylabel("Average Obesity Prevalence (%)")
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
plt.tight_layout()
plt.savefig("plot1_bmi_trend.png", dpi=150)
plt.show()

# Plot 2: Sex-wise BMI Trend
fig, ax = plt.subplots(figsize=(10, 5))
sex_colors = {'Both Sexes': COLORS['blue'], 'Male': COLORS['coral'], 'Female': COLORS['purple']}
for sex, grp in sex_yearly.groupby('Sex'):
    ax.plot(grp['Year'], grp['Avg_BMI_Percent'],
            label=sex, color=sex_colors.get(sex, 'gray'), linewidth=2)
ax.set_title("Obesity Prevalence Trend by Sex (Age 5-19)", fontsize=13, pad=12)
ax.set_xlabel("Year")
ax.set_ylabel("Average Obesity Prevalence (%)")
ax.legend(frameon=False)
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
plt.tight_layout()
plt.savefig("plot2_sex_trend.png", dpi=150)
plt.show()

# Plot 3: Top 10 Countries
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top10['Country'], top10['Avg_BMI_Percent'],
               color=COLORS['coral'], edgecolor='none')
for bar, val in zip(bars, top10['Avg_BMI_Percent']):
    ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va='center', fontsize=10)
ax.set_title("Top 10 Countries — Highest Childhood Obesity Prevalence", fontsize=13, pad=12)
ax.set_xlabel("Average Obesity Prevalence (%)")
ax.invert_yaxis()
ax.grid(axis='y', alpha=0)
plt.tight_layout()
plt.savefig("plot3_top10_countries.png", dpi=150)
plt.show()

# Plot 4: Bottom 10 Countries
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(bottom10['Country'], bottom10['Avg_BMI_Percent'],
               color=COLORS['green'], edgecolor='none')
for bar, val in zip(bars, bottom10['Avg_BMI_Percent']):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%", va='center', fontsize=10)
ax.set_title("Bottom 10 Countries — Lowest Childhood Obesity Prevalence", fontsize=13, pad=12)
ax.set_xlabel("Average Obesity Prevalence (%)")
ax.grid(axis='y', alpha=0)
plt.tight_layout()
plt.savefig("plot4_bottom10_countries.png", dpi=150)
plt.show()


# Plot 5: Boxplot by WHO Region
fig, ax = plt.subplots(figsize=(12, 6))
regions_ordered = (
    df_country[df_country['Sex'] == 'Both Sexes']
    .groupby('WHO_Region')['BMI_Percent']
    .median()
    .sort_values(ascending=False)
    .index.tolist()
)
data_by_region = [
    df_country[
        (df_country['WHO_Region'] == r) &
        (df_country['Sex'] == 'Both Sexes')
    ]['BMI_Percent'].dropna().values
    for r in regions_ordered
]
bp = ax.boxplot(data_by_region, patch_artist=True,
                medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], REGION_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_xticklabels(regions_ordered, rotation=15, ha='right')
ax.set_title("Obesity Prevalence Spread by WHO Region (Both Sexes)", fontsize=13, pad=12)
ax.set_ylabel("Obesity Prevalence (%)")
plt.tight_layout()
plt.savefig("plot6_boxplot_region.png", dpi=150)
plt.show()

# Plot 6: Regional Trend Over Years
fig, ax = plt.subplots(figsize=(11, 6))
for i, (region, grp) in enumerate(regional_yearly.groupby('WHO_Region')):
    ax.plot(grp['Year'], grp['Avg_BMI_Percent'],
            label=region, color=REGION_COLORS[i % len(REGION_COLORS)], linewidth=2)
ax.set_title("Obesity Prevalence Trend by WHO Region (1990-2022)", fontsize=13, pad=12)
ax.set_xlabel("Year")
ax.set_ylabel("Average Obesity Prevalence (%)")
ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
plt.tight_layout()
plt.savefig("plot7_regional_trend.png", dpi=150)
plt.show()

# Plot 7: Age Group Trend
fig, ax = plt.subplots(figsize=(10, 5))
age_colors = {
    '5-9 years':   COLORS['blue'],
    '10-19 years': COLORS['coral'],
    '5-19 years':  COLORS['green']
}
for age, grp in age_trend.groupby('Age_Group'):
    ax.plot(grp['Year'], grp['Avg_BMI_Percent'],
            label=age, color=age_colors.get(age, 'gray'),
            linewidth=2, linestyle='--' if age == '5-19 years' else '-')
ax.set_title("Obesity Prevalence by Age Group Over Years (Both Sexes)", fontsize=13, pad=12)
ax.set_xlabel("Year")
ax.set_ylabel("Average Obesity Prevalence (%)")
ax.legend(frameon=False)
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
plt.tight_layout()
plt.savefig("plot8_age_group_trend.png", dpi=150)
plt.show()


# Plot 8: Correlation Heatmap
fig, ax = plt.subplots(figsize=(7, 6))

# Renaming columns just for display purposes
corr_display = correlation.copy()
corr_display.columns = ['BMI %', 'CI Lower', 'CI Upper', 'Year']
corr_display.index   = ['BMI %', 'CI Lower', 'CI Upper', 'Year']

sns.heatmap(
    corr_display,
    annot=True, fmt=".2f", cmap="Blues",
    linewidths=0.5, linecolor='white',
    ax=ax, square=True,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 12}
)
ax.set_title("Correlation: BMI % vs Confidence Interval vs Year", fontsize=13, pad=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)  # horizontal x labels
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)  # horizontal y labels
plt.tight_layout()
plt.savefig("plot9_correlation_heatmap.png", dpi=150)
plt.show()

# Plot 9: Latest Year — Region x Sex Grouped Bar
latest_year = int(df_country['Year'].max())
latest = df_country[
    (df_country['Year'] == latest_year) &
    (df_country['Sex'].isin(['Male', 'Female'])) &
    (df_country['Age_Group'] == '5-19 years')
].copy()
pivot = (
    latest.groupby(['WHO_Region', 'Sex'])['BMI_Percent']
    .mean()
    .unstack()
    .reindex(columns=['Male', 'Female'])
)
fig, ax = plt.subplots(figsize=(11, 6))
pivot.plot(kind='bar', ax=ax,
           color=[COLORS['blue'], COLORS['pink']],
           edgecolor='none', width=0.6)
ax.set_title(f"Obesity Prevalence by Region & Sex in {latest_year} (Age 5-19)",
             fontsize=13, pad=12)
ax.set_xlabel("WHO Region")
ax.set_ylabel("Average Obesity Prevalence (%)")
ax.legend(frameon=False, title="Sex")
ax.set_xticklabels(pivot.index, rotation=20, ha='right')
plt.tight_layout()
plt.savefig("plot10_region_sex_bar.png", dpi=150)
plt.show()


# 14. SAVE CLEANED DATA

df_country.to_csv(
    r"C:\Users\monal\Downloads\cleaned_bmi_country_data.csv",
    index=False
)
print("\nEDA completed. All 10 plots saved and cleaned CSV exported.")
