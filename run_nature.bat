@echo off
echo Activating virtual environment...
call tf-env\Scripts\activate.bat

if %errorlevel% neq 0 (
    echo Error activating virtual environment.
    pause
    exit /b %errorlevel%
)

echo Setting up database...
mysql -u root -p"sqlpassword" -P portnumber < setup_database.sql

if %errorlevel% neq 0 (
    echo Error setting up database. Please check MySQL connection and credentials.
    pause
    exit /b %errorlevel%
)

echo Database setup successful.
echo Starting Flask server...
python app.py

pause 
