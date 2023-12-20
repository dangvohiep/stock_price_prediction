CREATE DATABASE sp500;

\c sp500

CREATE TABLE "company" (
    "symbol"    VARCHAR(10) PRIMARY KEY,
    "name"      VARCHAR(100),
    "gics"      VARCHAR(50)
);

CREATE TABLE "daily" (
    "date"      DATE NOT NULL,
    "symbol"    VARCHAR(10)     REFERENCES "company"("symbol") ON DELETE CASCADE,
    "open"      DECIMAL(15, 6)  CHECK ("open"   >= 0),
    "high"      DECIMAL(15, 6)  CHECK ("high"   >= 0),
    "low"       DECIMAL(15, 6)  CHECK ("low"    >= 0),
    "close"     DECIMAL(15, 6)  CHECK ("close"  >= 0),
    "volume"    BIGINT          CHECK ("volume" >= 0),
    PRIMARY KEY ("date", "symbol")
);

CREATE TABLE "prediction" (
    "id"        SERIAL PRIMARY KEY,
    "date"      DATE NOT NULL,
    "config"    JSONB NOT NULL,
    "symbol"    VARCHAR(10),
    "n_days_backward" INT CHECK ("n_days_backward" > 0),
    "n_days_forward"  INT CHECK ("n_days_forward" > 0),
    "value" JSONB NOT NULL,
    UNIQUE ("date", "config", "symbol", "n_days_backward", "n_days_forward")
);


