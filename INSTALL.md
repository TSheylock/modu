# Инструкция по установке

## Предварительные требования

1. **Node.js** (версия 16 или выше)
   - Скачать и установить с [официального сайта](https://nodejs.org/)
   - После установки перезагрузите компьютер

2. **MongoDB**
   - Скачать и установить [MongoDB Community Server](https://www.mongodb.com/try/download/community)
   - Создать директорию для данных: `C:\data\db`
   - Добавить MongoDB в PATH

3. **Redis для Windows**
   - Скачать и установить [Redis для Windows](https://github.com/microsoftarchive/redis/releases)
   - Добавить Redis в PATH

## Установка проекта

1. **Клонирование репозитория**
   ```bash
   git clone https://github.com/your-username/saske-ai.git
   cd saske-ai
   ```

2. **Установка зависимостей**
   ```bash
   npm install
   ```

3. **Настройка переменных окружения**
   - Скопируйте файл `.env.example` в `.env`
   - Настройте переменные окружения в файле `.env`

4. **Запуск MongoDB**
   ```bash
   mongod
   ```

5. **Запуск Redis**
   ```bash
   redis-server
   ```

6. **Запуск проекта**
   ```bash
   npm run dev
   ```

## Автоматическая установка

Для автоматической установки и настройки используйте скрипт `setup.ps1`:

```powershell
.\setup.ps1
```

## Проверка установки

После установки откройте в браузере:
- Frontend: http://localhost:3000
- Backend API: http://localhost:3001

## Возможные проблемы

1. **Ошибка "Node.js не установлен"**
   - Убедитесь, что Node.js установлен
   - Перезагрузите компьютер после установки
   - Проверьте переменную PATH

2. **Ошибка "MongoDB не установлен"**
   - Убедитесь, что MongoDB установлен
   - Создайте директорию для данных
   - Проверьте переменную PATH

3. **Ошибка "Redis не установлен"**
   - Убедитесь, что Redis установлен
   - Проверьте переменную PATH

4. **Ошибки при установке зависимостей**
   ```bash
   npm cache clean --force
   npm install
   ```

5. **Проблемы с портами**
   - Убедитесь, что порты 3000 и 3001 свободны
   - Измените порты в файле `.env` при необходимости

## Дополнительная информация

- [Документация Node.js](https://nodejs.org/docs)
- [Документация MongoDB](https://docs.mongodb.com)
- [Документация Redis](https://redis.io/documentation) 