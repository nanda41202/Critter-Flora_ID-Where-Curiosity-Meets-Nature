-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS nature_auth;

-- Use the database
USE nature_auth;

-- Create users table if it doesn't exist
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL
);
