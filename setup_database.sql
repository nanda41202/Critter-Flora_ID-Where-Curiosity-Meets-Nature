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

-- Insert or update the test user with hashed password
INSERT INTO users (email, password) 
VALUES ('loke@gmail.com', 'scrypt:32768:8:1$E5Fw8oF2hw5QpKkq$d0a66fa1d91a1320ea086838278898126853905066eba36dff53a94b70c2aa01b4a6d6c4c4bc2b3d91c23ac5cb2ec17cb5b16517a29af7f9a81fc473690a8a43')
ON DUPLICATE KEY UPDATE password = 'scrypt:32768:8:1$E5Fw8oF2hw5QpKkq$d0a66fa1d91a1320ea086838278898126853905066eba36dff53a94b70c2aa01b4a6d6c4c4bc2b3d91c23ac5cb2ec17cb5b16517a29af7f9a81fc473690a8a43'; 