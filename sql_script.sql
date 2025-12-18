CREATE DATABASE IF NOT EXISTS csat_db;
USE csat_db;

CREATE TABLE IF NOT EXISTS `user` (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS location (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    address TEXT,
    site_manager VARCHAR(255)
) ENGINE=InnoDB;

--CREATE TABLE IF NOT EXISTS employee (
--    id INT AUTO_INCREMENT PRIMARY KEY,
--    name VARCHAR(255) NOT NULL,
--    role VARCHAR(100),
--    experience_years INT
--) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS incident_type (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS incident (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date DATETIME NOT NULL,
    time VARCHAR(50),
    description TEXT,
    severity VARCHAR(50),
    weather VARCHAR(100),
    equipment_involved VARCHAR(255),
    cause TEXT,
    outcome TEXT,
    location_id INT,
    type_id INT,
    FOREIGN KEY (location_id) REFERENCES location(id) ON DELETE SET NULL,
    FOREIGN KEY (type_id) REFERENCES incident_type(id) ON DELETE SET NULL
) ENGINE=InnoDB;

--CREATE TABLE IF NOT EXISTS incident_employee (
--    incident_id INT,
--    employee_id INT,
--    PRIMARY KEY (incident_id, employee_id),
--    FOREIGN KEY (incident_id) REFERENCES incident(id) ON DELETE CASCADE,
--    FOREIGN KEY (employee_id) REFERENCES employee(id) ON DELETE CASCADE
--) ENGINE=InnoDB;

-- Note: The USER-INCIDENT relationship is not enforced via FK as it's a views/analyzes link, which may be handled in application logic.