-- ============================================================
-- SMART HSRP DATABASE SCHEMA
-- ============================================================

-- Users (auth)
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role          VARCHAR(50)  NOT NULL DEFAULT 'user',
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ============================================================
-- VIOLATIONS
-- Stores final gated violation records per vehicle track.
-- ============================================================
CREATE TABLE IF NOT EXISTS violations (
    id                      SERIAL PRIMARY KEY,

    -- Vehicle identification
    track_id                VARCHAR(64),
    vehicle_number          VARCHAR(20),      -- OCR plate number
    vehicle_class           VARCHAR(32),      -- car / motorcycle / bus / truck

    -- Violation details
    violation_type          VARCHAR(64),      -- non_hsrp_plate / no_helmet
    violation_confidence    FLOAT NOT NULL DEFAULT 0.0,

    -- Quality metrics
    stability_score         FLOAT NOT NULL DEFAULT 0.0,
    temporal_consistency    FLOAT NOT NULL DEFAULT 0.0,
    consecutive_detections  INTEGER NOT NULL DEFAULT 0,
    quality_score           FLOAT NOT NULL DEFAULT 0.0,  -- composite 0-1

    -- Review flags
    needs_manual_review     BOOLEAN NOT NULL DEFAULT FALSE,
    prediction_preceded     BOOLEAN NOT NULL DEFAULT FALSE,  -- was prediction fired?
    prediction_risk_max     FLOAT NOT NULL DEFAULT 0.0,

    -- Evidence frame range
    first_frame             INTEGER,
    last_frame              INTEGER,
    track_duration_frames   INTEGER,

    -- Media links (populated later)
    screenshot_path         VARCHAR(512),
    metadata_path           VARCHAR(512),

    -- Audit
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_violations_vehicle_number ON violations (vehicle_number);
CREATE INDEX IF NOT EXISTS idx_violations_violation_type ON violations (violation_type);
CREATE INDEX IF NOT EXISTS idx_violations_created_at    ON violations (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_violations_needs_review  ON violations (needs_manual_review);

-- ============================================================
-- ADAPTIVE THRESHOLDS
-- Persists learned thresholds per decision type.
-- ============================================================
CREATE TABLE IF NOT EXISTS adaptive_thresholds (
    decision_type       VARCHAR(64) PRIMARY KEY,
    current_threshold   FLOAT       NOT NULL DEFAULT 0.5,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed default rows so UI can read them even before any learning
INSERT INTO adaptive_thresholds (decision_type, current_threshold)
VALUES
    ('hsrp',           0.50),
    ('helmet',         0.40),
    ('ocr_confidence', 0.60)
ON CONFLICT (decision_type) DO NOTHING;

-- ============================================================
-- LEGACY TABLES (kept for backward compat)
-- ============================================================
CREATE TABLE IF NOT EXISTS events (
    id          SERIAL PRIMARY KEY,
    image_path  VARCHAR(512),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS vehicles (
    id                SERIAL PRIMARY KEY,
    event_id          INTEGER REFERENCES events(id) ON DELETE CASCADE,
    bbox_x1           INTEGER,
    bbox_y1           INTEGER,
    bbox_x2           INTEGER,
    bbox_y2           INTEGER,
    helmet_violation  BOOLEAN,
    helmet_confidence FLOAT,
    plate_text        VARCHAR(32),
    is_hsrp           BOOLEAN,
    hsrp_confidence   FLOAT,
    ocr_confidence    FLOAT
);
