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


# **Reference**
[Alex The Analyst](https://www.youtube.com/watch?v=4UltKCnnnTA&list=PLUaB-1hjhk8FE_XZ87vPPSfHqb6OcM0cF&index=19)

