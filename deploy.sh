#!/bin/bash

# Сборка проекта
npm run build

# Создание директории для сайта
sudo mkdir -p /var/www/saske.xyz/html

# Копирование файлов
sudo cp -r dist/* /var/www/saske.xyz/html/

# Установка прав
sudo chown -R www-data:www-data /var/www/saske.xyz
sudo chmod -R 755 /var/www/saske.xyz

# Перезапуск Nginx
sudo systemctl restart nginx

echo "Деплой завершен успешно!" 