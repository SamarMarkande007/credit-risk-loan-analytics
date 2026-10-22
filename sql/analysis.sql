-- =====================================================================
-- Credora Finance | 10 Analytical SQL Queries  (MySQL 8.0+)
-- Database: credora_risk   (run:  USE credora_risk;)
--
-- Ported from the PostgreSQL script in the PRD package. Syntax changes:
--   DATE_TRUNC('month', d)::date   -> DATE_FORMAT(d, '%Y-%m-01')
--   TO_CHAR(d, 'YYYY-MM')          -> DATE_FORMAT(d, '%Y-%m')
--   PERCENTILE_CONT(0.5) ...       -> window functions (MySQL has no median function)
--
-- Logic fixes (explained next to each query): 7.2 / 7.7 / 7.3
-- =====================================================================
USE credora_risk;

-- ---------------------------------------------------------------------
-- 7.1  CREDIT SCORE ANALYSIS: customers by credit-score band
--      Business question: how is the customer base spread across risk bands?
-- ---------------------------------------------------------------------
SELECT
    CASE
        WHEN credit_score >= 750 THEN 'Excellent (750-900)'
        WHEN credit_score >= 700 THEN 'Good (700-749)'
        WHEN credit_score >= 650 THEN 'Fair (650-699)'
        WHEN credit_score >= 580 THEN 'Poor (580-649)'
        ELSE 'Very Poor (300-579)'
    END                                       AS credit_band,
    COUNT(*)                                  AS customers,
    ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_customers,
    ROUND(AVG(credit_score), 0)               AS avg_score,
    ROUND(AVG(credit_utilization_ratio), 3)   AS avg_utilization,
    ROUND(AVG(num_late_payments_30d), 2)      AS avg_late_30d
FROM credit_history
GROUP BY credit_band
ORDER BY MIN(credit_score) DESC;


-- ---------------------------------------------------------------------
-- 7.2  LOAN APPROVAL TRENDS: monthly approval rate
--      FIX: the original divided by ALL applications, which counts
--      'Under Review' as if it were a rejection. The PRD's 46.4% approval
--      rate is measured on DECIDED applications only, so we do the same.
-- ---------------------------------------------------------------------
SELECT
    DATE_FORMAT(application_date, '%Y-%m-01')                          AS month,
    COUNT(*)                                                           AS applications,
    SUM(approval_status IN ('Approved', 'Rejected'))                   AS decided,
    SUM(approval_status = 'Approved')                                  AS approved,
    ROUND(100 * SUM(approval_status = 'Approved')
          / NULLIF(SUM(approval_status IN ('Approved', 'Rejected')), 0), 1) AS approval_rate_pct
FROM loan_applications
GROUP BY month
ORDER BY month;


-- ---------------------------------------------------------------------
-- 7.3  CUSTOMER SEGMENTATION: employment type x income band
--      FIX: the original LEFT JOIN let customers with no application count
--      as a 0 in the approval average. We now average over actual
--      applications only, and measure approval on decided applications.
-- ---------------------------------------------------------------------
SELECT
    c.employment_type,
    CASE
        WHEN c.annual_income >= 1500000 THEN '3. High (15L+)'
        WHEN c.annual_income >=  600000 THEN '2. Mid (6-15L)'
        ELSE                                 '1. Entry (<6L)'
    END                                                       AS income_band,
    COUNT(DISTINCT c.customer_id)                             AS customers,
    ROUND(AVG(ch.credit_score), 0)                            AS avg_credit_score,
    COUNT(la.application_id)                                  AS applications,
    ROUND(100 * SUM(la.approval_status = 'Approved')
          / NULLIF(SUM(la.approval_status IN ('Approved', 'Rejected')), 0), 1) AS approval_rate_pct
FROM customers c
JOIN credit_history ch          ON ch.customer_id = c.customer_id
LEFT JOIN loan_applications la  ON la.customer_id = c.customer_id
GROUP BY c.employment_type, income_band
ORDER BY c.employment_type, income_band;


-- ---------------------------------------------------------------------
-- 7.4  HIGH-RISK CUSTOMER IDENTIFICATION (rule-based watchlist, top 100)
--      Flags customers matching any bureau red flag, worst score first.
-- ---------------------------------------------------------------------
SELECT
    c.customer_id, c.first_name, c.last_name, c.city, c.state,
    ch.credit_score,
    ch.credit_utilization_ratio,
    ch.num_late_payments_90d,
    ch.num_defaults_prior
FROM customers c
JOIN credit_history ch ON ch.customer_id = c.customer_id
WHERE ch.credit_score < 600
   OR ch.credit_utilization_ratio > 0.80
   OR ch.num_defaults_prior >= 1
   OR ch.num_late_payments_90d >= 2
ORDER BY ch.credit_score ASC, ch.customer_id
LIMIT 100;


-- ---------------------------------------------------------------------
-- 7.5  DEFAULT RATE ANALYSIS by loan type (disbursed loans only)
-- ---------------------------------------------------------------------
SELECT
    loan_type,
    COUNT(*)                          AS disbursed_loans,
    SUM(loan_default)                 AS defaults,
    ROUND(100 * AVG(loan_default), 2) AS default_rate_pct,
    ROUND(AVG(interest_rate), 2)      AS avg_interest_rate
FROM loan_applications
WHERE disbursed_flag = 1 AND loan_default IS NOT NULL
GROUP BY loan_type
ORDER BY default_rate_pct DESC;


-- ---------------------------------------------------------------------
-- 7.6  LOAN AMOUNT DISTRIBUTION by loan type (incl. true median)
--      MySQL has no PERCENTILE_CONT, so the median is the average of the
--      middle one or two rows after ranking amounts within each loan type.
-- ---------------------------------------------------------------------
WITH ranked AS (
    SELECT
        loan_type,
        requested_amount,
        ROW_NUMBER() OVER (PARTITION BY loan_type ORDER BY requested_amount) AS rn,
        COUNT(*)     OVER (PARTITION BY loan_type)                           AS cnt
    FROM loan_applications
),
medians AS (
    SELECT loan_type, AVG(requested_amount) AS median_amount
    FROM ranked
    WHERE rn IN (FLOOR((cnt + 1) / 2), CEIL((cnt + 1) / 2))
    GROUP BY loan_type
)
SELECT
    r.loan_type,
    COUNT(*)                          AS applications,
    ROUND(MIN(r.requested_amount), 0) AS min_amount,
    ROUND(m.median_amount, 0)         AS median_amount,
    ROUND(AVG(r.requested_amount), 0) AS avg_amount,
    ROUND(MAX(r.requested_amount), 0) AS max_amount
FROM ranked r
JOIN medians m ON m.loan_type = r.loan_type
GROUP BY r.loan_type, m.median_amount
ORDER BY avg_amount DESC;


-- ---------------------------------------------------------------------
-- 7.7  MONTHLY APPROVAL STATISTICS with disbursed value
--      FIX: disbursed value now sums only loans with disbursed_flag = 1
--      (the original summed approved_amount for every row).
-- ---------------------------------------------------------------------
SELECT
    DATE_FORMAT(application_date, '%Y-%m')                              AS month,
    COUNT(*)                                                            AS applications,
    SUM(approval_status = 'Approved')                                   AS approved,
    SUM(approval_status = 'Rejected')                                   AS rejected,
    SUM(approval_status = 'Under Review')                               AS under_review,
    ROUND(SUM(CASE WHEN disbursed_flag = 1 THEN approved_amount ELSE 0 END) / 1000000, 2)
                                                                        AS disbursed_amount_mn
FROM loan_applications
GROUP BY month
ORDER BY month;


-- ---------------------------------------------------------------------
-- 7.8  RISK CATEGORISATION: blends bureau score and utilisation
--      (rule-based baseline; the ML risk score comes later in the project)
-- ---------------------------------------------------------------------
SELECT
    risk_category,
    COUNT(*)                                 AS customers,
    ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_customers,
    ROUND(AVG(credit_score), 0)              AS avg_score
FROM (
    SELECT
        customer_id,
        credit_score,
        CASE
            WHEN credit_score >= 700 AND credit_utilization_ratio < 0.4 THEN 'Low Risk'
            WHEN credit_score >= 620                                    THEN 'Medium Risk'
            ELSE 'High Risk'
        END AS risk_category
    FROM credit_history
) t
GROUP BY risk_category
ORDER BY MIN(credit_score) DESC;


-- ---------------------------------------------------------------------
-- 7.9  TRANSACTION BEHAVIOUR: 6-month cash-flow profile per customer
--      (top 50 by outflow). One row per customer = the grain we will use
--      when we join transactions to loans in feature engineering.
-- ---------------------------------------------------------------------
SELECT
    t.customer_id,
    COUNT(*)                                                         AS txn_count,
    ROUND(SUM(CASE WHEN transaction_type = 'Credit' THEN amount ELSE 0 END), 0) AS total_inflow,
    ROUND(SUM(CASE WHEN transaction_type = 'Debit'  THEN amount ELSE 0 END), 0) AS total_outflow,
    ROUND(SUM(CASE WHEN transaction_type = 'Credit' THEN amount ELSE -amount END), 0) AS net_cash_flow,
    ROUND(SUM(CASE WHEN category = 'EMI' THEN amount ELSE 0 END), 0) AS emi_outflow,
    ROUND(AVG(balance_after), 0)                                     AS avg_balance
FROM transactions t
GROUP BY t.customer_id
ORDER BY total_outflow DESC
LIMIT 50;


-- ---------------------------------------------------------------------
-- 7.10 TOP PREDICTORS OF LOAN DEFAULT: default rate for each risk driver
--      Compare each rate with the portfolio baseline (23.4%) in the last row.
-- ---------------------------------------------------------------------
SELECT driver, default_rate_pct, n
FROM (
    SELECT 'credit_score < 600' AS driver,
           ROUND(100 * AVG(la.loan_default), 2) AS default_rate_pct, COUNT(*) AS n
    FROM loan_applications la JOIN credit_history ch USING (customer_id)
    WHERE la.loan_default IS NOT NULL AND ch.credit_score < 600
    UNION ALL
    SELECT 'debt_to_income > 0.5',
           ROUND(100 * AVG(loan_default), 2), COUNT(*)
    FROM loan_applications
    WHERE loan_default IS NOT NULL AND debt_to_income_ratio > 0.5
    UNION ALL
    SELECT 'prior_defaults >= 1',
           ROUND(100 * AVG(la.loan_default), 2), COUNT(*)
    FROM loan_applications la JOIN credit_history ch USING (customer_id)
    WHERE la.loan_default IS NOT NULL AND ch.num_defaults_prior >= 1
    UNION ALL
    SELECT 'utilization > 0.8',
           ROUND(100 * AVG(la.loan_default), 2), COUNT(*)
    FROM loan_applications la JOIN credit_history ch USING (customer_id)
    WHERE la.loan_default IS NOT NULL AND ch.credit_utilization_ratio > 0.8
    UNION ALL
    SELECT 'ALL disbursed loans (baseline)',
           ROUND(100 * AVG(loan_default), 2), COUNT(*)
    FROM loan_applications
    WHERE loan_default IS NOT NULL
) d
ORDER BY default_rate_pct DESC;
