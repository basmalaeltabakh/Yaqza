-- backend/schema.sql
-- Reference only — الجداول بتتعمل من models.py

CREATE TABLE sensor_readings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    equipment_id TEXT    NOT NULL,
    cycle       INTEGER NOT NULL,
    setting1    REAL,
    setting2    REAL,
    setting3    REAL,
    sensor1     REAL,  sensor2  REAL,  sensor3  REAL,
    sensor4     REAL,  sensor5  REAL,  sensor6  REAL,
    sensor7     REAL,  sensor8  REAL,  sensor9  REAL,
    sensor10    REAL,  sensor11 REAL,  sensor12 REAL,
    sensor13    REAL,  sensor14 REAL,  sensor15 REAL,
    sensor16    REAL,  sensor17 REAL,  sensor18 REAL,
    sensor19    REAL,  sensor20 REAL,  sensor21 REAL,
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_equipment_id ON sensor_readings(equipment_id);
CREATE INDEX idx_cycle        ON sensor_readings(cycle);

CREATE TABLE predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    equipment_id TEXT    NOT NULL,
    rul          REAL    NOT NULL,
    failure_mode TEXT,
    confidence   REAL,
    timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pred_equipment ON predictions(equipment_id);