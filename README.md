# **Machine Learning Predict DIFOT Project using Python**
*This project using Python to Data Proceesing and run Machine Learning to predict DIFOT. A data set include: sales order, sale register,GPS_customer and GPS_distributor of top-leading FMCG industry company in VietNam.*

<img width="700" alt="image" src="https://github.com/user-attachments/assets/05591b19-ca09-4a0c-a7af-b91f32c14947">

# **Definition of DIFOT**
DIFOT stands for Delivery In Full, On Time. It is a key performance indicator (KPI) used in supply chain management to measure the percentage of deliveries that are both:
- In Full: The complete order was delivered without any shortages.
- On Time: The order was delivered on or before the agreed delivery date.

DIFOT is used to assess the reliability and efficiency of a company's delivery processes. It tracks how often customers receive their complete orders on the promised delivery date. A high DIFOT score indicates a high level of service reliability, while a low DIFOT score signals potential issues in the supply chain.

# **Python package using in project**
- Pandas/Numpy
- Pyspark/pyspark.sql
- Matplot/Seaborn
- Sklearn/xgboost/imbalanced-learn

# **Key takeaways**
- How to conduct full stages of machine learning project
- XGBoost Classifier is the best fit modeling for predict DIFOT
- Top 5 importance feature impact to DIFOT: distance_km,%urbanization rate, totalec_so, isfreegood_order_no, segmentation_Gold
- Actionable recomendations for 5 features above

# **Data Analysis Process**
<img width="700" alt="image" src="https://github.com/user-attachments/assets/494efa12-ba31-4e71-b370-b4a63740a1f3">

*Phases of project*

- Phase 1: Data Colection and Preprocessing
- Phase 2: EDA & Feature Engineering
- Phase 3: Model tranining & testing

## **Phase 1: Data Colection and Preprocessing**

### **Data colecting process**
<img width="700" alt="image" src="https://github.com/user-attachments/assets/6dace349-6bd2-4b24-ae40-3b3806d85f8d">

```
#Import table factdata show transaction of sales register & sales order  

from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col,to_date

factdata= spark.read.csv('/kaggle/working/DIFOT Result Full from May23 to Aug24 - Copy.csv')
header = factdata.first()
# Remove the first row from the DataFrame
factdata = factdata.filter(col("`_c0`") != header._c0) # Assuming _c0 is not null in the header
# Create a list of column names from the header row
columns = [header[i] for i in range(len(header))]
# Rename columns in the DataFrame
factdata = factdata.select([col(f"`_c{i}`").alias(name) for i, name in enumerate(columns)])

factdata = factdata.withColumn('order_date', to_date(col('order_date'), 'dd-MM-yyyy')) # Định dạng mẫu ví dụ
factdata = factdata.withColumn('settlement_date', to_date(col('settlement_date'), 'dd-MM-yyyy')) # Định dạng mẫu ví dụ
factdata= factdata.withColumn('totalec_sr', col('totalec_sr').cast(DoubleType()))
factdata= factdata.withColumn('totalNSR_sr', col('totalNSR_sr').cast(DoubleType()))
factdata= factdata.withColumn('totalNSR_so', col('totalNSR_so').cast(DoubleType()))
factdata= factdata.withColumn('totalec_so', col('totalec_so').cast(DoubleType()))
factdata= factdata.withColumn('diff_ec', col('diff_ec').cast(DoubleType()))
factdata= factdata.withColumn('diff_NSR', col('diff_NSR').cast(DoubleType()))
factdata.show(1)
factdata.printSchema()
```
<img width="400" alt="image" src="https://github.com/user-attachments/assets/9a8df9f9-3fcf-40c5-8396-f9fa9d82abdb">


### **Data Merging**

<img width="422" alt="image" src="https://github.com/user-attachments/assets/4caad72a-fb62-4514-b012-54b1e0f00774">

```
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

from pyspark.sql.functions import to_date, col

Listdayoff=spark.read.csv('/kaggle/working/List Day off .csv')
header3 = Listdayoff.first()
Listdayoff = Listdayoff.filter(col("`_c0`") != header3._c0)
columns = [header3[i] for i in range(len(header3))]
Listdayoff = Listdayoff.select([col(f"`_c{i}`").alias(name) for i, name in enumerate(columns)])

Listdayoff = Listdayoff.withColumn("d_date", to_date(col("d_date"), 'dd-MMM-yy'))
Listdayoff = Listdayoff.withColumn("eoweek KO", to_date(col("eoweek KO"), 'dd-MMM-yy'))
Listdayoff = Listdayoff.withColumn("eomonth KO", to_date(col("eomonth KO"), 'dd-MMM-yy'))
Listdayoff= Listdayoff.withColumn('year', col('year').cast(DoubleType()))

Listdayoff.show(2)

Listdayoff.printSchema()
```
### **Calculate new variables**
#### **Distance_km**

```
#1.Left join to get GPS location for Factdata table

from pyspark.sql.functions import col

# Join factdata and GPScus
factdata_with_GPScus = factdata.join(GPScus.alias('GPScus'), factdata.customer_code == GPScus.customerid, "left")


# Join factdata_with_GPScus and GPSdist
final_df = factdata_with_GPScus.join(GPSdist.alias('GPSdist'), factdata_with_GPScus.distributor_code == GPSdist.customerid, "left")

# Select col from DataFrame 
final_df = final_df.select(
    'order_date',
    'settlement_date',
    'region',
    'distributor_code',
    'customer_code',
    'segmentation',
    'channelname',
    'order_no',
    'item_code',
    'brand',
    'pack_type',
    'pack_size',
    'key_so',
    'totalec_sr',
    'totalNSR_sr',
    'totalNSR_so',
    'totalec_so',
    'diff_ec',
    'diff_NSR',
    col('GPScus.latitude').alias('GPScuslatitude'),
    col('GPScus.longitude').alias('GPScuslongitude'),
    col('GPSdist.latitude').alias('GPSdistlatitude'),
    col('GPSdist.longitude').alias('GPSdistlongitude')
)
#final_df.show()
print(final_df.columns)


#2.Calculate distance_km: Delivery Distance from Distributor (GPSdist) to Customer(GPScus)

import math
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

def haversine(lat1, lon1, lat2, lon2):
    # convert radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    r = 6371  # Radius of the Earth (km)
    return r * c

# Create UDF from function haversine
haversine_udf = udf(haversine, DoubleType())

final_df_with_distance = final_df.withColumn(
    'distance_km',
    haversine_udf(
        col('GPScuslatitude'),
        col('GPScuslongitude'),
        col('GPSdistlatitude'),
        col('GPSdistlongitude')
    )
)
# View DataFrame with distance_km col
print(final_df_with_distance.columns)

```
#### **Fill missing value**
```
#Fill Missing Value equal median


median_distance_km = 5.881606785112515

from pyspark.sql.functions import when

condition = (
    (col('GPScuslatitude') == 0) |
    (col('GPScuslongitude') == 0) |
    (col('GPSdistlatitude') == 0) |
    (col('GPSdistlongitude') == 0) |
    col('distance_km').isNull() |
    (col('distance_km') == 'N/A')
)


# Apply the condition to fill the distance_km column
final_df_with_distance=final_df_with_distance.cache()
final_df_filled = final_df_with_distance.withColumn(
    'distance_km',
    when(condition, median_distance_km).otherwise(col('distance_km'))
)

final_df_filled = final_df_filled.withColumn('order_date', to_date(col('order_date'), 'MM-dd-yyyy')) # Định dạng mẫu ví dụ
final_df_filled = final_df_filled.withColumn('settlement_date', to_date(col('settlement_date'), 'MM-dd-yyyy')) # Định dạng mẫu ví dụ

# Verify the results
final_df_filled.show(1)
```
#### **Filter Outlier**
```
#Remove outliers using IQR method

from pyspark.sql.functions import col
from pyspark.sql import SparkSession

# Spark session
spark = SparkSession.builder.appName("OutlierDetection").getOrCreate()

# Calculate (Q1 và Q3) and IQR
# Using approxQuantile
quantiles = final_df_filled.approxQuantile("distance_km", [0.25, 0.75], 0.01)

Q1 = quantiles[0]
Q3 = quantiles[1]
IQR = Q3 - Q1

# Calculate bound outlier
lower_bound = Q1 - 2 * IQR
upper_bound = Q3 + 2 * IQR

# Remove outliers
filtered_outliers_df = final_df_filled.filter(
    (col('distance_km') >= lower_bound) & (col('distance_km') <= upper_bound)
)

# Checking remaining data
total_rows = final_df_filled.count()
filtered_rows = filtered_outliers_df.count()
remaining_data_percentage = (filtered_rows / total_rows) * 100

print(f"IQR: {IQR}")
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")
print(f"Remaining Data Percentage: {remaining_data_percentage:.2f}%")
```
<img width="575" alt="image" src="https://github.com/user-attachments/assets/f10ae9c1-9d6e-4b49-83ae-51eafcda914f">

#### **Delivery Day**
```
from pyspark.sql.functions import col, datediff

filtered_outliers_df = filtered_outliers_df.withColumn(
    "DeliveryDay", datediff(col("settlement_date"), col("order_date"))
)

filtered_outliers_df = filtered_outliers_df.withColumn(
  "DeliveryDay", col("DeliveryDay").cast("double")
)

filtered_outliers_df.printSchema()

from pyspark.sql.functions import col, countDistinct, concat, lit, when, coalesce
from pyspark.sql import functions as F

# Step 1: Perform a left join between the two DataFrames based on the date range condition
joined_df = filtered_outliers_df.join(
    Listdayoff,
    (col("d_date") >= col("order_date")) & (col("d_date") <= col("settlement_date")),
    "left"
)

# Step 2: Create a new key by concatenating "order_date", "order_no", "settlement_date", and "customer_code" and group by this key
days_off_count_df = joined_df.withColumn(
    "order_key", concat(col("order_date").cast("string"), lit('-'), col("order_no"), lit('-'), col("settlement_date"), lit('-'), col("customer_code"))
).groupBy("order_key").agg(
    countDistinct("d_date").alias("days_off_count")
)

filtered_outliers_df.printSchema()
days_off_count_df.printSchema()

# Step 3: Join this result back to the original DataFrame using the new key and calculate Dayprocessing
filtered_outliers_df = filtered_outliers_df.withColumn(
    "order_key", concat(col("order_date").cast("string"), lit('-'), col("order_no"), lit('-'), col("settlement_date"), lit('-'), col("customer_code"))
).join(
    days_off_count_df, "order_key", "left"
).withColumn(
    "days_off_count", coalesce(days_off_count_df["days_off_count"], lit(0))
).withColumn(
    "Dayprocessing", when(col("DeliveryDay") - col("days_off_count") < 0, lit(0))
                    .otherwise(col("DeliveryDay") - col("days_off_count"))
)

# Create 'Ontime'column
filtered_outliers_df = filtered_outliers_df.withColumn(
    "Ontime", when(col("Dayprocessing") <= 2, "Yes").otherwise("No")
)

# Show the resulting DataFrame with the new column
filtered_outliers_df.show(1)

# Print the schema of the resulting DataFrame
filtered_outliers_df.printSchema()

```
<img width="500" alt="image" src="https://github.com/user-attachments/assets/5bfa3ab6-831d-4718-95bc-8fbaa20bbdbc">

#### **Infull and DIFOT**
```
filtered_outliers_df.cache()
filtered_outliers_df = filtered_outliers_df.withColumn(
    "Infull", when(col("diff_ec") <= 0.1, "Yes").otherwise("No")
)
filtered_outliers_df = filtered_outliers_df.withColumn(
    "DIFOT", when((col("Infull") == "Yes") & (col("Ontime") == "Yes"), "Yes").otherwise("No")
)

filtered_outliers_df.printSchema()
```
<img width="277" alt="image" src="https://github.com/user-attachments/assets/f1dd94ea-c748-4945-aedb-9676cc7ed0c5">

## **Phase 2: EDA & Feature Engineering**
### **Visualization**

```
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
import plotly.express as px

from pyspark.sql import functions as F
import numpy as np
from pyspark.sql.functions import month, year
import seaborn as sns
import matplotlib.pyplot as plt


# Tạo bảng với số lượng đơn hàng khác nhau khi DIFOT = 0 và DIFOT = 1

summary_df = filtered_outliers_df.groupBy("customer_code","GPScuslongitude","GPScuslatitude") \
    .agg(
        F.countDistinct(F.when(F.col('DIFOT') == 0, F.col('order_no'))).alias('order_non_difot'),
        F.countDistinct(F.when(F.col('DIFOT') == 1, F.col('order_no'))).alias('order_difot')
    )

# Thêm cột % DIFOT (tính theo phần trăm)
summary_df = summary_df.withColumn(
    '%DIFOT',
    (F.col('order_difot') / (F.col('order_non_difot') + F.col('order_difot')) * 100)
)
summary_df = summary_df.toPandas()
summary_df = summary_df.dropna(subset=["GPScuslongitude","GPScuslatitude"])
df_geo = gpd.GeoDataFrame (summary_df, geometry = gpd.points_from_xy(
summary_df.GPScuslongitude, summary_df.GPScuslatitude))
# Filter out rows where GPScuslatitude or GPScuslongitude is 0
df_geo['GPScuslatitude'] = pd.to_numeric(df_geo['GPScuslatitude'], errors='coerce')
df_geo['GPScuslongitude'] = pd.to_numeric(df_geo['GPScuslongitude'], errors='coerce')

df_geo = df_geo[(df_geo['GPScuslatitude'] >= 1) & (df_geo['GPScuslongitude'] >= 1)]
import geopandas as gpd
import matplotlib.pyplot as plt

# Load the world map data
world_data = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Filter for Asia and then further for Vietnam
axis = world_data[(world_data.continent == 'Asia') & (world_data.name == 'Vietnam')].plot(
    color='lightblue', edgecolor='black')

# Plot the customer locations
df_geo.plot(
    ax=axis, 
    marker='o', 
    column='%DIFOT', 
    markersize=df_geo['%DIFOT'],  # Point size will be proportional to %DIFOT
    cmap='coolwarm',  # Optional color map to highlight values
    legend=True,  # Optional legend for %DIFOT
    alpha=0.6,  # Optional transparency
    edgecolor='k'
)

# Add title and set figure size
plt.title('Customer Distribution in Vietnam Based on % DIFOT', fontsize=15)
fig = plt.gcf()
fig.set_size_inches(9, 6)

# Save and show the figure
fig.savefig('customer_distribution.png', dpi=200)
plt.show()

import pandas as pd
from io import StringIO

# Add in GPS of 63 Province of Viet Nam
data_str = """
Thành phố Hồ Chí Minh,10.8333,106.63278
Hà Nội,21.0333,105.85000
Hải Phòng,20.86194,106.68028
Cần Thơ,10.03278,105.78389
Đà Nẵng,16.05194,108.21528
Biên Hòa,10.95694,106.84306
Thanh Hoa,19.80750,105.77639
Nha Trang,12.23889,109.19694
Vũng Tàu,10.34583,107.08472
Thủ Đức,10.88333,106.72694
Huế,16.46278,107.58472
Buôn Ma Thuột,12.66667,108.03889
Thái Nguyên,21.56750,105.82556
Vinh,18.68083,105.68167
Hải Dương,20.93972,106.31250
Thủ Dầu Một,10.99333,106.65611
Nam Định,20.42000,106.16833
Rạch Giá,10.02083,105.09028
Hạ Long,20.97194,107.04528
Mỹ Tho,10.35417,106.36528
Quy Nhơn,13.77500,109.23333
Thái Bình,20.44750,106.33750
Đà Lạt,11.94556,108.44222
Phan Thiết,10.92222,108.10944
Cà Mau,9.18361,105.15000
Long Thành,10.79306,107.01361
Tuy Hòa,13.08222,109.31611
Cẩm Phả,21.01611,107.33194
Pleiku,13.98361,108.00000
Sóc Trăng,9.60389,105.97417
Phù Mỹ,14.21639,109.11694
Long Xuyên,10.37528,105.41833
Tây Ninh,11.36778,106.11917
Bảo Lộc,11.53056,107.77806
Bắc Ninh,21.18528,106.05639
Bạc Liêu,9.25889,105.75194
Chí Linh,21.16194,106.41806
Lạng Sơn,21.85417,106.76167
Vĩnh Long,10.24583,105.95833
Trà Vinh,9.95139,106.33472
Bến Lức,10.63194,106.49306
Bắc Giang,21.29139,106.18694
Tân An,10.53111,106.41250
Hưng Yên,20.63667,106.05694
Cam Ranh,11.91361,109.13694
Đồng Hới,17.46861,106.59944
Phú Quốc,10.29611,103.98667
Ninh Bình,20.25111,105.97500
Bình Minh,10.02917,105.85250
Lào Cai,22.44028,104.00278
Phan Rang – Tháp Chàm,11.56667,108.99167
Hoài Ân,14.30444,108.85528
Yên Bái,21.71667,104.89861
Vĩnh Yên,21.29889,105.60611
Móng Cái,21.53333,107.96667
Sa Đéc,10.30000,105.76667
Tam Kỳ,15.56528,108.49444
Kon Tum,14.35000,107.99861
Cao Bằng,22.66694,106.26028
Điện Biên Phủ,21.36667,103.00861
Tuyên Quang,21.81861,105.21167
Quảng Ngãi,15.12389,108.81167
Hội An,15.88056,108.33889
Hà Tĩnh,18.34083,105.90750
Sơn La,21.32722,103.91417
Đảo Côn Lôn,8.69306,106.60944
Thang Binh District,15.68944,108.38000
Quảng Trị,16.74694,107.19389
Gia Lâm,21.03333,105.95889
Cam Lâm,12.07556,109.14028
"""

# Using StringIO to convert file-like object
data_io = StringIO(data_str)

# Create DataFrame From String
data_df = pd.read_csv(data_io, header=None, names=["City", "Latitude", "Longitude"])


def haversine(lon1, lat1, lon2, lat2):
    # Convert latitude and longitude from degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    
    # Earth radius in kilometers
    r = 6371  
    return c * r

def assign_province(customer_lat, customer_lon): 
    province_lons = data_df['Longitude'].values
    province_lats = data_df['Latitude'].values
    distances = haversine(province_lons, province_lats, customer_lon, customer_lat)
    nearest_index = np.argmin(distances)  # Get the index of the nearest province
    
    return data_df.loc[nearest_index, 'City']  # Find nearest province

# Assign the nearest province to each customer
df_geo['nearest_tinh'] = df_geo.apply(lambda row: assign_province(row['GPScuslatitude'], row['GPScuslongitude']), axis=1)

# Group data by the nearest province in df_geo
df_grouped = df_geo.groupby('nearest_tinh').agg({'%DIFOT': 'mean', 'GPScuslatitude': 'first', 'GPScuslongitude': 'first'}).reset_index()



import matplotlib.pyplot as plt

# Create a plot with a specified figure size

import geopandas as gpd
import matplotlib.pyplot as plt

# Load the world map data
world_data = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Filter for Asia and then further for Vietnam
df_grouped = gpd.GeoDataFrame (df_grouped, geometry = gpd.points_from_xy(
df_grouped.GPScuslongitude, df_grouped.GPScuslatitude))

axis = world_data[(world_data.continent == 'Asia') & (world_data.name == 'Vietnam')].plot(
    color='lightblue', edgecolor='black')


df_grouped.plot(
    ax=axis, 
    marker='o', 
    column='%DIFOT', 
    markersize=df_grouped['%DIFOT']*2,  # Point size will be proportional to %DIFOT
    cmap='coolwarm',  # Optional color map to highlight values
    legend=True,  # Optional legend for %DIFOT
    alpha=0.6,  # Optional transparency
    edgecolor='k'
)

# Add title and set figure size
plt.title('%DIFOT Distribution in 63 Pronvinces of Vietnam', fontsize=15)
fig = plt.gcf()
fig.set_size_inches(12, 6)

# Save and show the figure
fig.savefig('customer_distribution.png', dpi=500)
plt.figure(figsize=(20, 8))
plt.show()
```
![output](https://github.com/user-attachments/assets/218b3d0a-261c-46fc-8098-ae29360a1980)

```
import seaborn as sns
import matplotlib.pyplot as plt

# Group by region and % urbanization rate, and calculate the mean of DIFOT

summary_df = filtered_outliers_df.groupBy('% urbanization rate') \
    .agg(
        F.countDistinct(F.when(F.col('DIFOT') == 0, F.col('order_no'))).alias('order_non_difot'),
        F.countDistinct(F.when(F.col('DIFOT') == 1, F.col('order_no'))).alias('order_difot')
    )

summary_df = summary_df.withColumn(
    '%DIFOT',
    (F.col('order_difot') / (F.col('order_non_difot') + F.col('order_difot')) * 100)
)
summary_df =summary_df.toPandas()

# Create the scatter plot
sns.scatterplot(data=summary_df , x='% urbanization rate', y='%DIFOT')

# Add plot labels and title
plt.xlabel('% Urbanization Rate')
plt.ylabel('DIFOT(%)')
plt.title('DIFOT vs Urbanization Rate')
#rangey_chart=list(range(0,100,5))
# plt.ylim(50,100,5)
plt.xlim(0,100,5)

# Show the plot
plt.show()
```
![output1](https://github.com/user-attachments/assets/1fbc51c1-f5bc-4cb9-b8aa-0ec0d04c83a0)

```
from pyspark.sql import functions as F
import numpy as np


# Create table have number of order_difot and order_non_difot
summary_df = filtered_outliers_df.groupBy('region', 'item_code') \
    .agg(
        F.countDistinct(F.when(F.col('DIFOT') == 0, F.col('order_no'))).alias('order_non_difot'),
        F.countDistinct(F.when(F.col('DIFOT') == 1, F.col('order_no'))).alias('order_difot')
    )

# Calculate % DIFOT
summary_df = summary_df.withColumn(
    '%DIFOT',
    (F.col('order_difot') / (F.col('order_non_difot') + F.col('order_difot')) * 100)
)
summary_df=summary_df.toPandas()
# create pivot table %DIFOT
pivot_heat = pd.pivot_table(
    summary_df,
    index='item_code',        
    columns='region',         
    values='%DIFOT'           
)
# Create col VN equal sum of all Regions
pivot_heat['VN'] = pivot_heat.sum(axis=1)

# Sort ASC by col VN & Show top 20
pivot_heat_top20 = pivot_heat.sort_values(by='VN', ascending=False).head(20)
pivot_heat_top20 = pivot_heat_top20.drop(columns=['VN'])
region_mapping = {
    'Central  Region': 'CP',
    'Hanoi Region': 'HN',
    'Ho Chi Minh Region': 'HC',
    'Mekong Delta Region': 'MK',
    'North East Region': 'NE',
    'North West Region': 'NW',
    'South Provinces Reg': 'SP'
}

pivot_heat_top20.rename(columns=region_mapping, inplace=True)


annot_data = pivot_heat_top20.applymap(lambda x: '{:.1f}%'.format(x))

# Create heatmap using annot format %
sns.heatmap(data=pivot_heat_top20,
            annot=annot_data,  #
            fmt="",  
            cmap="Blues", 
            linecolor='white', linewidths=0.5)


ax.set_yticklabels(yticks, rotation=0);
ax.set_xticklabels(xticks, rotation=90);
title1='TOP 20 ITEM CODE WITH HIGHEST % DIFOT'
ax.set_title(title1,loc='center',fontsize=18)

sns.heatmap
```
![output2](https://github.com/user-attachments/assets/b8863ddb-d337-4ea3-b8d0-3c6448facd8b)

### **Feature engineering**
#### **Correlation matrix**

```
spark.conf.set("spark.sql.execution.arrow.enabled", "false")
# filtered_outliers_df.printSchema()
corre_df = filtered_outliers_df.sample(fraction=0.1,seed=1505).select(
    'totalNSR_so', 'totalec_so', 'distance_km', 'DeliveryDay', 'days_off_count', 
    'Dayprocessing', 'Ontime', 'Infull', 'DIFOT', '% urbanization rate','isfreegood_order_no'
)
corre_df=corre_df.toPandas()
corre_matrix=corre_df.corr()
corre_matrix

import numpy as np
ones_corr=np.ones_like(corre_matrix,dtype=bool)
ones_corr.shape , corre_matrix.shape 
mask=np.triu(ones_corr)
adjustedmask=mask[1:, :-1]
adjusted_corre_matrix=corre_matrix.iloc[1:, :-1]

fig, ax=plt.subplots(figsize=(10,8))
sns.heatmap(data=adjusted_corre_matrix,mask=adjustedmask,
            annot=True,fmt=".2f",cmap="Blues",vmin=-1,vmax=1,
           linecolor='white', linewidths=0.5)

yticks=[i.upper() for i in adjusted_corre_matrix.index]

xticks=[i.upper() for i in adjusted_corre_matrix.columns]

ax.set_yticklabels(yticks, rotation=0);
ax.set_xticklabels(xticks, rotation=90);
title='CORRELATION MAXTRIX OF DIFOT VARIABLE'
ax.set_title(title,loc='center',fontsize=18)
```
<img width="600" alt="image" src="https://github.com/user-attachments/assets/1579a6ae-e37d-4cec-b9d6-f93cc22c0820">

#### **Feature importance**

```
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import matplotlib.pyplot as plt

# numerical variables
numeric_var = ['totalec_so', 'distance_km', 'DeliveryDay', 'days_off_count', 'Dayprocessing', '% urbanization rate', 'isfreegood_order_no']
X_numeric = X[numeric_var]  # Dữ liệu đầu vào X

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier

# categorical variables
categorical_var = ['region', 'channelname', 'segmentation', 'brand', 'pack_type', 'pack_size', 'month']

# One-Hot Encoding categorical var
encoder = OneHotEncoder(sparse=False, drop='first')
X_encoded = encoder.fit_transform(X[categorical_var])

X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_var))
finalfeatureselection=pd.concat([X[numeric_var].reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)
finalfeatureselection

#  Modeling ExtraTreesClassifier
model = ExtraTreesClassifier(random_state=1505)

# Training model
model.fit(finalfeatureselection, Y)

# print result top 15 feature
print(model.feature_importances_)  
feat_importances2 = pd.Series(model.feature_importances_, index=finalfeatureselection.columns)
feat_importances2.nlargest(15).plot(kind='barh', color='skyblue')
plt.title('Top 15 Important Features')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.show()
feat_importances2
```
<img width="431" alt="image" src="https://github.com/user-attachments/assets/37847bba-2de8-489c-89c6-830a46d9d803">

## **Phase 3: Model tranining & testing**
### **Handling imbalance data**
```
# Random Extract 10% dataset for Machine Learning Stage (Due to Data was too large)

import pandas as pd

X=filtered_outliers_df.sample(fraction= 0.1,seed=1505).select(
    'totalec_so', 'distance_km', 'DeliveryDay', 'days_off_count', 
    'Dayprocessing', '% urbanization rate','isfreegood_order_no','order_date',
    'region','segmentation', 'channelname','brand','pack_type','pack_size', 
)
X=X.toPandas()

# format datetime col order_date 
X['order_date'] = pd.to_datetime(X['order_date'])

# Create col month
X['month'] = X['order_date'].dt.month

#Create X and Y for ML Stage
X=X.values
import numpy as np
X = np.delete(X, 7, axis=1)

print(X.shape)

Y=filtered_outliers_df.sample(fraction= 0.1,seed=1505).select('DIFOT')
Y=Y.toPandas()

# Rename 'channelname' to 'segmentation' 
X.rename(columns={'channelname': 'temp_channelname', 'segmentation': 'channelname'}, inplace=True)
X.rename(columns={'temp_channelname': 'segmentation'}, inplace=True)

X['channelname'].fillna('DRINKING', inplace=True)
X['segmentation'].fillna('Bronze', inplace=True)
X['brand'].fillna('*******', inplace=True)
X['pack_type'].fillna('PET', inplace=True)
X['pack_size'].fillna('320ML',inplace=True)

X_final=finalfeatureselection[['Dayprocessing','DeliveryDay','totalec_so','distance_km','days_off_count','% urbanization rate','isfreegood_order_no','month_3','pack_size_297ML','month_4','month_8','region_North West Region','pack_size_300ML','branding','segmentation_Gold']]
# Using undersampling NearMiss to solve imbalance dataset

X_final.shape,Y.shape
from imblearn.under_sampling import NearMiss
nm = NearMiss()

X_res, y_res = nm.fit_resample(X_final,Y)

X_res.shape, y_res.shape
```
<img width="456" alt="image" src="https://github.com/user-attachments/assets/b3126792-de69-4ead-bfe1-88c01c7980b8">

### **Data splitting**
```
from sklearn.model_selection import train_test_split
np.random.seed(1505)
X_train,X_test,Y_train,Y_test =train_test_split(X_res,y_res,test_size=0.2 )
```
<img width="462" alt="image" src="https://github.com/user-attachments/assets/fc861aaa-a05e-4e4a-a085-b2bf4c030f48">

### **Training model**
```
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Create XGB Model
model = XGBClassifier(use_label_encoder=False,eval_metrics='logloss',random_state=1505)
k = 5
scores = cross_val_score(model, X_train, Y_train, cv=k, scoring='accuracy')

for fold, score in enumerate(scores, start=1):
    print(f'Accuracy for fold {fold}: {score:.4f}')

mean_accuracy = np.mean(scores)
std_accuracy = np.std(scores)
print(f'\nMean Accuracy (K-Fold = {k}): {mean_accuracy:.4f} ± {std_accuracy:.4f}')

#.... The same with another models
```
<img width="338" alt="image" src="https://github.com/user-attachments/assets/79dd7890-eefc-4d1e-8a23-ed72c1313226">

## **Testing model**
### **Evaluation model**

```
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def print_single_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# Example usage with your DecisionTreeClassifier model
dt_model=XGBClassifier(n_estimators=200,use_label_encoder=False,eval_metrics='logloss',random_state=1505)

# Train the model
dt_model.fit(X_train, Y_train)

# Predict test data
y_pred = dt_model.predict(X_test)

# Print the single representative scores
print_single_score(Y_test, y_pred)

# ........ Do the same with another models
```
<img width="345" alt="image" src="https://github.com/user-attachments/assets/a91021ba-ed9f-4e57-a6a2-636bc187b346">

#### *Model selection XGB*
```
# List top 5 feature in XGB model
# Choosing metric weight
importance = dt_model.get_booster().get_score(importance_type='weight')
importance_df = pd.DataFrame(importance.items(), columns=['Feature', 'Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)
top_10_features = importance_df.head(5)
print(top_10_features)

# Visual top 5 feature
top_10_features.plot(kind='barh', x='Feature', y='Importance', legend=False)
plt.title('Top 5 Feature Importances')
plt.show()
```
<img width="400" alt="image" src="https://github.com/user-attachments/assets/a7e45467-d92e-4cd2-be7d-45db43bc4e0e">

```
# Demo one tree in XGB classifier model 

import matplotlib.pyplot as plt
from xgboost import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(dt_model, num_trees=9, rankdir='LR')
plt.show()

# List out features in tree
booster = dt_model.get_booster()
feature_names = booster.feature_names

tree = booster.get_dump()[9]  

# Print features
print("Nội dung cây thứ 10:")
print(tree)

features_in_tree = set()

for line in tree.split('\n'):
    if 'feature' in line:
        parts = line.split(' ')
        for part in parts:
            if 'feature' in part:
                feature_index = int(part.split('[')[-1].replace(']', ''))
                features_in_tree.add(feature_names[feature_index])

print("Feature in tree number 10:")
for feature in features_in_tree:
    print(feature)
```
<img width="586" alt="image" src="https://github.com/user-attachments/assets/c3f2efdd-0575-49f6-8258-ee582ea27b06">

#### *Checking underfitting and overfitting*
```
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluating in traningset
    train_accuracy = accuracy_score(Y_train, y_train_pred)
    train_precision = precision_score(Y_train, y_train_pred)
    train_recall = recall_score(Y_train, y_train_pred)
    train_f1 = f1_score(Y_train, y_train_pred)
    
    # Evaluating in testset
    test_accuracy = accuracy_score(Y_test, y_test_pred)
    test_precision = precision_score(Y_test, y_test_pred)
    test_recall = recall_score(Y_test, y_test_pred)
    test_f1 = f1_score(Y_test, y_test_pred)
    
    # Print result
    print("Training Performance:")
    print(f"Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-Score: {train_f1:.4f}")
    print("\nTest Performance:")
    print(f"Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1-Score: {test_f1:.4f}")
    
    # Compare testset and trainingset
    print("\nModel Evaluation:")
    if train_accuracy > test_accuracy + 0.05:
        print("The model is likely overfitting.")
    elif train_accuracy < test_accuracy - 0.05:
        print("The model is likely underfitting.")
    else:
        print("The model is performing well without signs of overfitting or underfitting.")
        

from sklearn.tree import DecisionTreeClassifier
dt_model=XGBClassifier(use_label_encoder=False,eval_metrics='logloss',random_state=1505)

evaluate_model(dt_model, X_train, Y_train, X_test, Y_test)
```
<img width="547" alt="image" src="https://github.com/user-attachments/assets/cad1b857-1a63-4a02-92ca-d292a332e3f0">

# **Actionable recomendations for 5 features above**

- Delivery Distance (distance_km).
Delivery distance significantly impacts on-time performance. To improve DIFOT, the company should optimize routes using GPS, analyze common routes, and select appropriate vehicles. Establishing distribution centers closer to customers and partnering with local delivery services can also reduce distance and improve delivery speed.

- Urbanization Rate (%urbanization rate).
Urbanization affects delivery efficiency. To enhance DIFOT, the company should study urbanization levels in different areas and adjust delivery strategies accordingly. This may include offering express delivery services or setting up delivery points in densely populated areas to reduce delivery times.

- Total Order Quantity (totalec_so).
Order quantity impacts transport efficiency. To boost DIFOT, the company can encourage bulk orders through promotions, offering discounts or free shipping for larger orders. This reduces transportation costs and optimizes delivery operations, enhancing overall efficiency.

- Promotional Goods Availability (isfreegood_order_no).
Promotional items attract customers and drive sales. To improve DIFOT, the company should ensure promotional stock availability through tight inventory management. Clear communication about promotions can increase customer loyalty and satisfaction, enhancing DIFOT performance.

- Gold Customer Segment (segmentation_Gold).
Gold customers are high-value clients requiring premium service. To improve DIFOT for this segment, the company should offer priority delivery, quality checks before shipping, and dedicated customer service. This increases satisfaction and loyalty, contributing to a higher DIFOT.


# **Thanks For Reading !!!**


