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

# **Phase 1: Data Proceesing**
## **Remove duplicate**
```sql
-- Check duplicates (row_num > 1)
WITH dup_cte AS
(
SELECT *,
ROW_NUMBER() OVER(
PARTITION BY company,location,industry,total_laid_off,percentage_laid_off,'date',stage,funds_raised_millions) as row_num
FROM layoffs_cleaned
)
SELECT *
FROM dup_cte
WHERE row_num > 1;

--Create a new table before delete duplicate
CREATE TABLE `layoffs_cleaned_ver2` (
  `company` text,
  `location` text,
  `industry` text,
  `total_laid_off` int DEFAULT NULL,
  `percentage_laid_off` text,
  `date` text,
  `stage` text,
  `country` text,
  `funds_raised_millions` int DEFAULT NULL,
  `row_num` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--Add value into table was deleted duplicate (row_num = 1)
INSERT layoffs_cleaned_ver2
WITH dup_cte AS
(
SELECT *,
ROW_NUMBER() OVER(
PARTITION BY company,location,industry,total_laid_off,percentage_laid_off,'date',stage,funds_raised_millions) as row_num
FROM layoffs_cleaned
)
SELECT *
FROM dup_cte
WHERE row_num = 1;

```
<img width="700" alt="duplicate_table" src="https://github.com/user-attachments/assets/b9139eed-f603-4632-99cb-c6c8c35b9bd9">

## **Standardize Data**

```
--industry column

SELECT DISTINCT industry
FROM layoffs_cleaned_ver2
ORDER BY 1;

SELECT DISTINCT industry
FROM layoffs_cleaned_ver2
WHERE industry LIKE 'Crypto%'
ORDER BY 1;

UPDATE layoffs_cleaned_ver2
SET industry ='Crypto'
WHERE industry LIKE 'Crypto%';
```
<img width="700" alt="image" src="https://github.com/user-attachments/assets/02759931-32b1-4bb8-9b58-df3cdd918679">

```
--country column
SELECT DISTINCT country
FROM layoffs_cleaned_ver2
ORDER BY 1;

SELECT DISTINCT country
FROM layoffs_cleaned_ver2
WHERE country LIKE 'United States%';

UPDATE layoffs_cleaned_ver2
SET country = TRIM(TRAILING '.' FROM country);
```

<img width="700" alt="image" src="https://github.com/user-attachments/assets/96ff3e33-6f9b-44bb-aaad-9dbbde29d62f">

```
--date column
SELECT date, STR_TO_DATE(date, '%m/%d/%Y') as date_trans
FROM layoffs_cleaned_ver2;

UPDATE layoffs_cleaned_ver2
SET date = STR_TO_DATE(date, '%m/%d/%Y');
```

<img width="700" alt="image" src="https://github.com/user-attachments/assets/66901976-a1ad-43e1-9c99-1677ad248246">

## **Fill Null and BLank values**

```
-- Industry column

-- Check industry is Null 
SELECT *
FROM layoffs_cleaned_ver2
WHERE industry IS NULL OR industry LIKE '';

UPDATE layoffs_cleaned_ver2
SET industry = NULL
WHERE industry = '';

-- Check all company missing value in industry column
SELECT *
FROM layoffs_cleaned_ver2 lo1 	
JOIN layoffs_cleaned_ver2 lo2
ON lo1.company = lo2.company 
WHERE (lo1.industry IS NULL OR lo1.industry ='') 
AND lo2.industry IS NOT NULL;

-- Update missing industry value for all company
UPDATE layoffs_cleaned_ver2 lo1
JOIN layoffs_cleaned_ver2 lo2
ON lo1.company = lo2.company 
SET lo1.industry=lo2.industry
WHERE lo1.industry IS NULL AND lo2.industry IS NOT NULL;
```
<img width="700" alt="image" src="https://github.com/user-attachments/assets/416771bb-d214-433c-8937-32a7d0f759db">

```
-- total_laid_off Column and percentage_laid_off Column

SELECT *
FROM layoffs_cleaned_ver2
WHERE total_laid_off IS NULL AND percentage_laid_off IS NULL;

-- Delete rows have null in total_laid_off and percentage_laid_off
DELETE FROM layoffs_cleaned_ver2
WHERE total_laid_off IS NULL AND percentage_laid_off IS NULL;
```

<img width="700" alt="image" src="https://github.com/user-attachments/assets/d86a15b7-e72b-4f05-a0fa-0cbc985f7148">


## **Remove Any Columns**

```
SELECT *
FROM layoffs_cleaned_ver2;

ALTER TABLE layoffs_cleaned_ver2
DROP row_num;
```

# **Phase 2: Exploratory Data Analysis (EDA)**

## **Group by categorical variables**

```
-- Industry
SELECT industry, SUM(total_laid_off) as total_off
FROM layoffs_cleaned_ver2
WHERE industry IS NOT NULL
GROUP BY industry
ORDER BY 2 DESC;

-- Country
SELECT country, SUM(total_laid_off) as total_off
FROM layoffs_cleaned_ver2
WHERE country IS NOT NULL
GROUP BY country 
ORDER BY 2 DESC;

-- Stage
SELECT stage, SUM(total_laid_off) as total_off
FROM layoffs_cleaned_ver2
WHERE stage IS NOT NULL
GROUP BY stage
ORDER BY 2 DESC;
```
<img width="700" alt="image" src="https://github.com/user-attachments/assets/59f11cab-8d2a-48fa-9c4b-8e300eaf066d">

## **Time serries analysis**

```
-- Total laid off by each year
SELECT YEAR(date) as YEAR, SUM(total_laid_off) as total_off
FROM layoffs_cleaned_ver2
WHERE YEAR(date) IS NOT NULL
GROUP BY YEAR(date)
ORDER BY 1 DESC;
```
<img width="700" alt="image" src="https://github.com/user-attachments/assets/40939b4e-2c1d-489d-910f-50d571378364">


```
--  Constribution of months in full year 2022 
SELECT 
    MONTH, 
    total_off_month,
	(total_off_month / total_off_2022) * 100 AS percentage_of_month,
    total_off_2022
FROM (
    SELECT 
        MONTH(date) AS MONTH, 
        SUM(total_laid_off) AS total_off_month,
        (SELECT SUM(total_laid_off) 
         FROM layoffs_cleaned_ver2 
         WHERE YEAR(date) IN (2022)) AS total_off_2022
    FROM layoffs_cleaned_ver2
    WHERE YEAR(date) IN (2022)
    GROUP BY MONTH(date)
    ORDER BY 1 ASC
) monthconstribution;

```
<img width="700" alt="image" src="https://github.com/user-attachments/assets/bf1fdeba-0495-4f7d-a3f7-ed9cb3b75889">

```
-- Acumulative total laid off by month-year
SELECT date_time, total_off , SUM(total_off) OVER (ORDER BY date_time) as rooling_total
FROM(
SELECT SUBSTRING(date,1,7) as date_time, SUM(total_laid_off) as total_off
FROM layoffs_cleaned_ver2
WHERE SUBSTRING(date,1,7) IS NOT NULL
GROUP BY SUBSTRING(date,1,7)
ORDER BY 1 ASC
) timeanalysis;
```
<img width="700" alt="image" src="https://github.com/user-attachments/assets/d4dc2d4c-d909-4971-b7ae-db680e0c7f1e">

## **Advance analysis using CTE**
```
-- Top 5 company have highest laid off in each year
WITH company_year AS
( SELECT company, YEAR(date) as years, SUM(total_laid_off) as total_off
FROM layoffs_cleaned_ver2
GROUP BY company,YEAR(date)
), company_year_rank AS
(SELECT *, DENSE_RANK() OVER (PARTITION BY years ORDER BY total_off DESC) as ranking
FROM company_year
WHERE total_off IS NOT NULL AND years IS NOT NULL
)
SELECT *
FROM company_year_rank
WHERE ranking <=5
;
```
<img width="700" alt="image" src="https://github.com/user-attachments/assets/70606910-a18f-4588-964f-5bb8ea30e31f">

# **Reference**
[Alex The Analyst](https://www.youtube.com/watch?v=4UltKCnnnTA&list=PLUaB-1hjhk8FE_XZ87vPPSfHqb6OcM0cF&index=19)

