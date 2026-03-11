# SQL Query Library (Few-Shot Examples)

### Example 1: Comparing Internal Sales (Orders) with IMS Benchmarks
```sql
WITH Internal AS (
  SELECT ib."name", SUM(ms."product_quantity") as qty 
  FROM "master_sale" ms 
  JOIN "customer_details" cd ON ms."customer_id" = cd."customer_id" 
  JOIN "ims_brick" ib ON cd."ims_brick_id" = ib."id" 
  GROUP BY 1
),
Market AS (
  SELECT ib."name", SUM("unit") as qty 
  FROM "ims_sale" s 
  JOIN "ims_brick" ib ON s."brickId" = ib."id" 
  GROUP BY 1
)
SELECT 
  COALESCE(i."name", m."name") AS "Brick Name", 
  COALESCE(i.qty, 0) AS "Internal Units", 
  COALESCE(m.qty, 0) AS "IMS Market Units"
FROM Internal i 
FULL OUTER JOIN Market m ON i."name" = m."name"
ORDER BY "Internal Units" DESC
LIMIT 5;
```

### Example 2: Manager Visit Frequency
```sql
SELECT 
  m."name", 
  COUNT(dp."id") AS "Visit Count"
FROM "managers" AS m
JOIN "doctor_plan" AS dp ON m."id" = dp."managerId"
GROUP BY m."name"
ORDER BY "Visit Count" DESC;
```

### Example 3: Finding Doctors by Specialization and Region
```sql
SELECT 
  dr."name", 
  sp."name" AS "Specialization", 
  r."name" AS "Region"
FROM "doctors" AS dr
JOIN "specializatons" AS sp ON dr."id" = ANY(SELECT "B" FROM "_doctorsTospecializatons" WHERE "A" = dr."id") -- Handling many-to-many
JOIN "healthcentres" AS hc ON dr."id" = ANY(SELECT "A" FROM "_doctorsTohealthcentres" WHERE "B" = dr."id")
JOIN "regions" AS r ON hc."region_id" = r."id";
```
