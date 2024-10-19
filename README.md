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



# **Reference**
[Alex The Analyst](https://www.youtube.com/watch?v=4UltKCnnnTA&list=PLUaB-1hjhk8FE_XZ87vPPSfHqb6OcM0cF&index=19)

