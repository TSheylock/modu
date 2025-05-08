# Проверка и установка необходимых компонентов
Write-Host "Проверка и установка необходимых компонентов..." -ForegroundColor Green

# Проверка Node.js
$nodeVersion = node --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Node.js не установлен. Пожалуйста, установите Node.js с https://nodejs.org/" -ForegroundColor Red
    Write-Host "После установки Node.js перезапустите этот скрипт" -ForegroundColor Yellow
    exit
}
Write-Host "Node.js установлен: $nodeVersion" -ForegroundColor Green

# Проверка npm
$npmVersion = npm --version
Write-Host "npm установлен: $npmVersion" -ForegroundColor Green

# Проверка MongoDB
$mongoVersion = mongod --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "MongoDB не установлен. Пожалуйста, установите MongoDB с https://www.mongodb.com/try/download/community" -ForegroundColor Red
    Write-Host "После установки MongoDB перезапустите этот скрипт" -ForegroundColor Yellow
    exit
}
Write-Host "MongoDB установлен: $mongoVersion" -ForegroundColor Green

# Проверка Redis
$redisVersion = redis-cli --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Redis не установлен. Пожалуйста, установите Redis с https://github.com/microsoftarchive/redis/releases" -ForegroundColor Red
    Write-Host "После установки Redis перезапустите этот скрипт" -ForegroundColor Yellow
    exit
}
Write-Host "Redis установлен: $redisVersion" -ForegroundColor Green

# Установка зависимостей проекта
Write-Host "Установка зависимостей проекта..." -ForegroundColor Green
npm install

# Проверка переменных окружения
if (-not (Test-Path .env)) {
    Write-Host "Файл .env не найден. Создаем из примера..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "Пожалуйста, настройте переменные окружения в файле .env" -ForegroundColor Yellow
}

# Запуск MongoDB
Write-Host "Запуск MongoDB..." -ForegroundColor Green
Start-Process mongod

# Запуск Redis
Write-Host "Запуск Redis..." -ForegroundColor Green
Start-Process redis-server

# Запуск проекта
Write-Host "Запуск проекта..." -ForegroundColor Green
npm run dev 