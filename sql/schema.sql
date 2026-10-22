-- =====================================================================
-- Credora Finance | Credit Risk & Loan Approval Analytics Platform
-- Schema for MySQL 8.0.16+  (CHECK constraints are only enforced from 8.0.16)
--
-- Ported from the PostgreSQL script in the PRD package:
--   * DROP ... CASCADE      -> FOREIGN_KEY_CHECKS toggle + DROP TABLE
--   * NUMERIC / SMALLINT    -> DECIMAL / SMALLINT (same meaning)
--   * email UNIQUE removed  -> the source data contains 2 duplicate emails
--                              (email is excluded from modelling per the PRD,
--                               so we keep a plain index instead)
--
-- Run it either with:   python src/data_load.py          (recommended)
-- or manually in Workbench:  CREATE DATABASE credora_risk; USE credora_risk; then run this file.
-- =====================================================================

-- ---------------------------------------------------------------------
-- 0. Clean start (safe re-run)
-- ---------------------------------------------------------------------
SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS loan_applications;
DROP TABLE IF EXISTS credit_history;
DROP TABLE IF EXISTS customers;
SET FOREIGN_KEY_CHECKS = 1;

-- ---------------------------------------------------------------------
-- 1. CUSTOMERS (parent table)
-- ---------------------------------------------------------------------
CREATE TABLE customers (
    customer_id         VARCHAR(12)    NOT NULL,
    first_name          VARCHAR(50)    NOT NULL,
    last_name           VARCHAR(50)    NOT NULL,
    gender              VARCHAR(10),
    date_of_birth       DATE           NOT NULL,
    age                 SMALLINT,
    marital_status      VARCHAR(15),
    dependents          SMALLINT       DEFAULT 0,
    education           VARCHAR(30),
    employment_type     VARCHAR(20),
    occupation          VARCHAR(60),
    annual_income       DECIMAL(12,2),
    city                VARCHAR(50),
    state               VARCHAR(50),
    residence_type      VARCHAR(20),
    years_at_residence  SMALLINT,
    email               VARCHAR(120),
    phone               VARCHAR(20),
    kyc_status          VARCHAR(15)    DEFAULT 'Pending',
    customer_since      DATE           NOT NULL,
    PRIMARY KEY (customer_id),
    CONSTRAINT chk_cust_gender   CHECK (gender IN ('Male','Female','Other')),
    CONSTRAINT chk_cust_age      CHECK (age BETWEEN 18 AND 100),
    CONSTRAINT chk_cust_deps     CHECK (dependents >= 0),
    CONSTRAINT chk_cust_income   CHECK (annual_income >= 0),
    CONSTRAINT chk_cust_years    CHECK (years_at_residence >= 0),
    CONSTRAINT chk_cust_kyc      CHECK (kyc_status IN ('Verified','Pending','Rejected'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------
-- 2. CREDIT_HISTORY (1:1 with customer, bureau snapshot)
-- ---------------------------------------------------------------------
CREATE TABLE credit_history (
    credit_id                  VARCHAR(12)   NOT NULL,
    customer_id                VARCHAR(12)   NOT NULL,
    as_of_date                 DATE          NOT NULL,
    credit_score               SMALLINT      NOT NULL,
    num_open_accounts          SMALLINT,
    num_credit_inquiries_6m    SMALLINT,
    credit_utilization_ratio   DECIMAL(4,3),
    num_late_payments_30d      SMALLINT      DEFAULT 0,
    num_late_payments_90d      SMALLINT      DEFAULT 0,
    num_defaults_prior         SMALLINT      DEFAULT 0,
    bankruptcies               SMALLINT      DEFAULT 0,
    total_credit_limit         DECIMAL(14,2),
    total_outstanding_debt     DECIMAL(14,2),
    oldest_account_age_months  SMALLINT,
    PRIMARY KEY (credit_id),
    UNIQUE KEY uq_credit_customer (customer_id),          -- enforces the 1:1 relationship
    CONSTRAINT fk_credit_customer
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id) ON DELETE CASCADE,
    CONSTRAINT chk_credit_score  CHECK (credit_score BETWEEN 300 AND 900),
    CONSTRAINT chk_credit_util   CHECK (credit_utilization_ratio BETWEEN 0 AND 1),
    CONSTRAINT chk_credit_counts CHECK (num_open_accounts >= 0 AND num_credit_inquiries_6m >= 0
                                        AND num_late_payments_30d >= 0 AND num_late_payments_90d >= 0
                                        AND num_defaults_prior >= 0 AND bankruptcies >= 0),
    CONSTRAINT chk_credit_amounts CHECK (total_credit_limit >= 0 AND total_outstanding_debt >= 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------
-- 3. LOAN_APPLICATIONS (many:1 with customer, holds the ML target)
-- ---------------------------------------------------------------------
CREATE TABLE loan_applications (
    application_id            VARCHAR(12)   NOT NULL,
    customer_id               VARCHAR(12)   NOT NULL,
    application_date          DATE          NOT NULL,
    loan_type                 VARCHAR(20)   NOT NULL,
    loan_purpose              VARCHAR(50),
    requested_amount          DECIMAL(14,2),
    loan_term_months          SMALLINT,
    interest_rate             DECIMAL(5,2),
    applicant_monthly_income  DECIMAL(12,2),
    existing_emi              DECIMAL(12,2) DEFAULT 0,
    debt_to_income_ratio      DECIMAL(7,3),
    collateral_flag           SMALLINT      DEFAULT 0,
    approval_status           VARCHAR(15)   NOT NULL,
    approved_amount           DECIMAL(14,2) DEFAULT 0,
    decision_date             DATE,
    disbursed_flag            SMALLINT      DEFAULT 0,
    loan_default              SMALLINT,                    -- ML target, NULL when not disbursed
    PRIMARY KEY (application_id),
    CONSTRAINT fk_loan_customer
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id) ON DELETE CASCADE,
    CONSTRAINT chk_loan_amount   CHECK (requested_amount > 0),
    CONSTRAINT chk_loan_term     CHECK (loan_term_months > 0),
    CONSTRAINT chk_loan_rate     CHECK (interest_rate >= 0),
    CONSTRAINT chk_loan_income   CHECK (applicant_monthly_income >= 0),
    CONSTRAINT chk_loan_dti      CHECK (debt_to_income_ratio >= 0),
    CONSTRAINT chk_loan_collat   CHECK (collateral_flag IN (0,1)),
    CONSTRAINT chk_loan_status   CHECK (approval_status IN ('Approved','Rejected','Under Review')),
    CONSTRAINT chk_loan_disb     CHECK (disbursed_flag IN (0,1)),
    CONSTRAINT chk_loan_default  CHECK (loan_default IN (0,1)),
    CONSTRAINT chk_decision_after_application
        CHECK (decision_date IS NULL OR decision_date >= application_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------
-- 4. TRANSACTIONS (many:1 with customer, behavioural signal)
-- ---------------------------------------------------------------------
CREATE TABLE transactions (
    transaction_id    VARCHAR(14)   NOT NULL,
    customer_id       VARCHAR(12)   NOT NULL,
    transaction_date  DATE          NOT NULL,
    transaction_type  VARCHAR(10),
    category          VARCHAR(30),
    channel           VARCHAR(20),
    amount            DECIMAL(14,2),
    balance_after     DECIMAL(16,2),
    PRIMARY KEY (transaction_id),
    CONSTRAINT fk_txn_customer
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id) ON DELETE CASCADE,
    CONSTRAINT chk_txn_type   CHECK (transaction_type IN ('Credit','Debit')),
    CONSTRAINT chk_txn_amount CHECK (amount >= 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------
-- 5. INDEXES (query-path optimisation for analytics and joins)
--    MySQL already indexes PKs, FKs and the UNIQUE key above, so we only
--    add the extra analytical ones.
-- ---------------------------------------------------------------------
CREATE INDEX idx_cust_email       ON customers (email);
CREATE INDEX idx_cust_state       ON customers (state);
CREATE INDEX idx_credit_score     ON credit_history (credit_score);
CREATE INDEX idx_loan_status      ON loan_applications (approval_status);
CREATE INDEX idx_loan_appdate     ON loan_applications (application_date);
CREATE INDEX idx_loan_default     ON loan_applications (loan_default);
CREATE INDEX idx_loan_type        ON loan_applications (loan_type);
CREATE INDEX idx_txn_date         ON transactions (transaction_date);
CREATE INDEX idx_txn_category     ON transactions (category);
