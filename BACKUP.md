# Руководство по резервному копированию

## Обзор

SASKE.ai использует комплексный подход к резервному копированию, включающий:

1. Резервное копирование базы данных
2. Резервное копирование файлов
3. Резервное копирование конфигураций
4. Резервное копирование блокчейна
5. Восстановление из резервных копий

## Резервное копирование базы данных

### MongoDB

```bash
# Скрипт резервного копирования MongoDB
#!/bin/bash

# Конфигурация
BACKUP_DIR="/backup/mongodb"
DATE=$(date +%Y%m%d_%H%M%S)
MONGODB_URI="mongodb://localhost:27017"
DATABASES=("saske" "auth" "analytics")

# Создание директории для бэкапа
mkdir -p "$BACKUP_DIR/$DATE"

# Бэкап каждой базы данных
for DB in "${DATABASES[@]}"
do
  mongodump \
    --uri="$MONGODB_URI" \
    --db="$DB" \
    --out="$BACKUP_DIR/$DATE"
done

# Сжатие бэкапа
tar -czf "$BACKUP_DIR/$DATE.tar.gz" "$BACKUP_DIR/$DATE"

# Удаление несжатого бэкапа
rm -rf "$BACKUP_DIR/$DATE"

# Удаление старых бэкапов (оставляем последние 7 дней)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
```

### Redis

```bash
# Скрипт резервного копирования Redis
#!/bin/bash

# Конфигурация
BACKUP_DIR="/backup/redis"
DATE=$(date +%Y%m%d_%H%M%S)
REDIS_HOST="localhost"
REDIS_PORT="6379"

# Создание директории для бэкапа
mkdir -p "$BACKUP_DIR/$DATE"

# Бэкап Redis
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" SAVE
cp /var/lib/redis/dump.rdb "$BACKUP_DIR/$DATE/"

# Сжатие бэкапа
tar -czf "$BACKUP_DIR/$DATE.tar.gz" "$BACKUP_DIR/$DATE"

# Удаление несжатого бэкапа
rm -rf "$BACKUP_DIR/$DATE"

# Удаление старых бэкапов (оставляем последние 7 дней)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
```

## Резервное копирование файлов

### S3

```typescript
// Конфигурация S3
const s3Config = {
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
  }
};

// Функция бэкапа в S3
const backupToS3 = async (filePath: string, bucket: string, key: string) => {
  const s3 = new AWS.S3(s3Config);
  
  const fileStream = fs.createReadStream(filePath);
  
  await s3.upload({
    Bucket: bucket,
    Key: key,
    Body: fileStream
  }).promise();
};
```

### Локальное хранилище

```bash
# Скрипт резервного копирования файлов
#!/bin/bash

# Конфигурация
BACKUP_DIR="/backup/files"
DATE=$(date +%Y%m%d_%H%M%S)
SOURCE_DIRS=(
  "/data/uploads"
  "/data/exports"
  "/data/temp"
)

# Создание директории для бэкапа
mkdir -p "$BACKUP_DIR/$DATE"

# Бэкап каждой директории
for DIR in "${SOURCE_DIRS[@]}"
do
  rsync -av --delete "$DIR" "$BACKUP_DIR/$DATE/"
done

# Сжатие бэкапа
tar -czf "$BACKUP_DIR/$DATE.tar.gz" "$BACKUP_DIR/$DATE"

# Удаление несжатого бэкапа
rm -rf "$BACKUP_DIR/$DATE"

# Удаление старых бэкапов (оставляем последние 30 дней)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete
```

## Резервное копирование конфигураций

### Kubernetes

```bash
# Скрипт резервного копирования Kubernetes
#!/bin/bash

# Конфигурация
BACKUP_DIR="/backup/kubernetes"
DATE=$(date +%Y%m%d_%H%M%S)
NAMESPACES=("production" "staging" "development")

# Создание директории для бэкапа
mkdir -p "$BACKUP_DIR/$DATE"

# Бэкап каждого namespace
for NS in "${NAMESPACES[@]}"
do
  # Бэкап всех ресурсов
  kubectl get all -n "$NS" -o yaml > "$BACKUP_DIR/$DATE/$NS-all.yaml"
  
  # Бэкап ConfigMaps
  kubectl get configmap -n "$NS" -o yaml > "$BACKUP_DIR/$DATE/$NS-configmaps.yaml"
  
  # Бэкап Secrets
  kubectl get secret -n "$NS" -o yaml > "$BACKUP_DIR/$DATE/$NS-secrets.yaml"
done

# Сжатие бэкапа
tar -czf "$BACKUP_DIR/$DATE.tar.gz" "$BACKUP_DIR/$DATE"

# Удаление несжатого бэкапа
rm -rf "$BACKUP_DIR/$DATE"

# Удаление старых бэкапов (оставляем последние 30 дней)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete
```

## Резервное копирование блокчейна

### Ethereum

```bash
# Скрипт резервного копирования Ethereum
#!/bin/bash

# Конфигурация
BACKUP_DIR="/backup/ethereum"
DATE=$(date +%Y%m%d_%H%M%S)
DATA_DIR="/var/lib/ethereum"

# Создание директории для бэкапа
mkdir -p "$BACKUP_DIR/$DATE"

# Остановка ноды
systemctl stop ethereum

# Бэкап данных
tar -czf "$BACKUP_DIR/$DATE.tar.gz" "$DATA_DIR"

# Запуск ноды
systemctl start ethereum

# Удаление старых бэкапов (оставляем последние 7 дней)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
```

## Восстановление из резервных копий

### MongoDB

```bash
# Скрипт восстановления MongoDB
#!/bin/bash

# Конфигурация
BACKUP_FILE="$1"
MONGODB_URI="mongodb://localhost:27017"

# Проверка наличия файла
if [ ! -f "$BACKUP_FILE" ]; then
  echo "Backup file not found: $BACKUP_FILE"
  exit 1
fi

# Распаковка бэкапа
tar -xzf "$BACKUP_FILE" -C /tmp

# Восстановление каждой базы данных
for DB in /tmp/*/
do
  DB_NAME=$(basename "$DB")
  mongorestore \
    --uri="$MONGODB_URI" \
    --db="$DB_NAME" \
    "$DB"
done

# Очистка
rm -rf /tmp/*
```

### Redis

```bash
# Скрипт восстановления Redis
#!/bin/bash

# Конфигурация
BACKUP_FILE="$1"
REDIS_HOST="localhost"
REDIS_PORT="6379"

# Проверка наличия файла
if [ ! -f "$BACKUP_FILE" ]; then
  echo "Backup file not found: $BACKUP_FILE"
  exit 1
fi

# Остановка Redis
systemctl stop redis

# Распаковка бэкапа
tar -xzf "$BACKUP_FILE" -C /tmp

# Копирование файла
cp /tmp/dump.rdb /var/lib/redis/

# Запуск Redis
systemctl start redis

# Очистка
rm -rf /tmp/*
```

## Мониторинг резервного копирования

### Метрики

```typescript
// Конфигурация метрик
const backupMetrics = {
  size: new Gauge({
    name: 'saske_backup_size_bytes',
    help: 'Size of backup in bytes'
  }),
  duration: new Histogram({
    name: 'saske_backup_duration_seconds',
    help: 'Duration of backup in seconds'
  }),
  status: new Gauge({
    name: 'saske_backup_status',
    help: 'Status of backup (1: success, 0: failure)'
  })
};
```

### Алерты

```yaml
# Правила алертов
groups:
- name: backup
  rules:
  - alert: BackupFailed
    expr: saske_backup_status == 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: Backup failed
      description: Backup process failed for 5 minutes

  - alert: BackupTooLarge
    expr: saske_backup_size_bytes > 1e9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: Backup too large
      description: Backup size is above 1GB
```

## Лучшие практики

### Планирование

1. Определите критичные данные
2. Установите частоту бэкапов
3. Выберите хранилище
4. Настройте ротацию

### Выполнение

1. Используйте автоматизацию
2. Проверяйте целостность
3. Мониторьте процесс
4. Документируйте процедуры

### Хранение

1. Используйте разные локации
2. Шифруйте данные
3. Контролируйте доступ
4. Регулярно проверяйте

### Восстановление

1. Регулярно тестируйте
2. Документируйте процедуры
3. Тренируйте команду
4. Поддерживайте актуальность

## Ресурсы

- [MongoDB Backup Documentation](https://docs.mongodb.com/manual/core/backup/)
- [Redis Backup Documentation](https://redis.io/topics/persistence)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [Kubernetes Backup Tools](https://kubernetes.io/docs/tasks/administer-cluster/backup-restore/) 