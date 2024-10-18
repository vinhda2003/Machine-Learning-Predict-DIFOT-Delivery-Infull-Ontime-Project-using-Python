-- Requirement: Using my SQL to data cleaning and Exploratory Data Analysis (EDA) 
-- Dataset Info
SELECT 
    (SELECT COUNT(*) FROM layoffs) AS number_of_rows, 
    (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE table_name = 'layoffs') AS number_of_columns;

-- I. Data Cleaning:
-- 1. Remove duplicates
-- 2. Standardize the Data
-- 3. Fill Null and BLank values
-- 4. Remove Any Columns

SELECT *
FROM layoffs;

-- Create a new table using for data cleaning

CREATE TABLE layoffs_cleaned
LIKE layoffs;

INSERT layoffs_cleaned
SELECT *
FROM layoffs;

SELECT *
FROM layoffs_cleaned;

-- 1. Remove duplicates
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

SELECT *
FROM layoffs_cleaned_ver2;

-- 2. Standardize Data

-- Industry column
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

-- Country column
SELECT DISTINCT country
FROM layoffs_cleaned_ver2
ORDER BY 1;

SELECT *
FROM layoffs_cleaned_ver2
WHERE country LIKE 'United States%';

UPDATE layoffs_cleaned_ver2
SET country ='United States'
WHERE country LIKE 'United States%';

UPDATE layoffs_cleaned_ver2
SET country = TRIM(TRAILING '.' FROM country)
;

-- Date column
SELECT date, STR_TO_DATE(date, '%m/%d/%Y') as date_trans
FROM layoffs_cleaned_ver2;

UPDATE layoffs_cleaned_ver2
SET date = STR_TO_DATE(date, '%m/%d/%Y');


-- 3. Fill Null and BLank values

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



-- total_laid_off Column and percentage_laid_off Column

SELECT *
FROM layoffs_cleaned_ver2
WHERE total_laid_off IS NULL AND percentage_laid_off IS NULL;

-- Delete rows have null in total_laid_off and percentage_laid_off
DELETE FROM layoffs_cleaned_ver2
WHERE total_laid_off IS NULL AND percentage_laid_off IS NULL;


-- 4. Remove Any Columns

SELECT *
FROM layoffs_cleaned_ver2;

ALTER TABLE layoffs_cleaned_ver2
DROP row_num;

-- II. Exploratory Data Analysis (EDA) 
-- 1. Group by categorical variables
-- 2. Time serries analysis
-- 3. Advance analysis using CTE

SELECT *
FROM layoffs_cleaned_ver2;

-- 1. Group by categorical variables

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


-- 2. Time serries analysis


-- Total laid off by each year
SELECT YEAR(date) as YEAR, SUM(total_laid_off) as total_off
FROM layoffs_cleaned_ver2
WHERE YEAR(date) IS NOT NULL
GROUP BY YEAR(date)
ORDER BY 1 DESC;


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

-- Acumulative total laid off by month-year

SELECT date_time, total_off , SUM(total_off) OVER (ORDER BY date_time) as rooling_total
FROM(
SELECT SUBSTRING(date,1,7) as date_time, SUM(total_laid_off) as total_off
FROM layoffs_cleaned_ver2
WHERE SUBSTRING(date,1,7) IS NOT NULL
GROUP BY SUBSTRING(date,1,7)
ORDER BY 1 ASC
) timeanalysis;


-- 3. Advance analysis using CTE
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
