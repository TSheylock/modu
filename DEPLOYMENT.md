# Руководство по развертыванию

## Обзор

SASKE.ai использует многоуровневый подход к развертыванию, включающий:

1. Развертывание в разработке
2. Развертывание в тестировании
3. Развертывание в продакшене
4. Мониторинг развертывания
5. Откат развертывания

## Развертывание в разработке

### Docker Compose

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    volumes:
      - .:/app
      - /app/node_modules
    environment:
      - NODE_ENV=development
      - REACT_APP_API_URL=http://localhost:3001

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.dev
    ports:
      - "3001:3001"
    volumes:
      - ./backend:/app
      - /app/node_modules
    environment:
      - NODE_ENV=development
      - MONGODB_URI=mongodb://mongodb:27017/saske
      - REDIS_URL=redis://redis:6379

  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  mongodb_data:
  redis_data:
```

### Локальное развертывание

```bash
# Скрипт развертывания в разработке
#!/bin/bash

# Остановка существующих контейнеров
docker-compose -f docker-compose.dev.yml down

# Сборка и запуск контейнеров
docker-compose -f docker-compose.dev.yml up --build -d

# Ожидание готовности сервисов
echo "Waiting for services to be ready..."
sleep 10

# Проверка статуса
docker-compose -f docker-compose.dev.yml ps
```

## Развертывание в тестировании

### Kubernetes

```yaml
# k8s/staging/frontend.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: saske-frontend
  namespace: staging
spec:
  replicas: 2
  selector:
    matchLabels:
      app: saske-frontend
  template:
    metadata:
      labels:
        app: saske-frontend
    spec:
      containers:
      - name: saske-frontend
        image: saske-ai:staging
        ports:
        - containerPort: 3000
        env:
        - name: NODE_ENV
          value: "staging"
        - name: REACT_APP_API_URL
          value: "http://saske-backend:3001"
---
apiVersion: v1
kind: Service
metadata:
  name: saske-frontend
  namespace: staging
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 3000
  selector:
    app: saske-frontend
```

### CI/CD Pipeline

```yaml
# .github/workflows/staging.yml
name: Staging Deployment

on:
  push:
    branches:
      - develop

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v1
      
    - name: Login to DockerHub
      uses: docker/login-action@v1
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
        
    - name: Build and push
      uses: docker/build-push-action@v2
      with:
        context: .
        push: true
        tags: saske-ai:staging
        
    - name: Deploy to Kubernetes
      uses: azure/k8s-deploy@v1
      with:
        manifests: |
          k8s/staging/*.yaml
        images: |
          saske-ai:staging
        namespace: staging
```

## Развертывание в продакшене

### Kubernetes

```yaml
# k8s/production/frontend.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: saske-frontend
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: saske-frontend
  template:
    metadata:
      labels:
        app: saske-frontend
    spec:
      containers:
      - name: saske-frontend
        image: saske-ai:production
        ports:
        - containerPort: 3000
        env:
        - name: NODE_ENV
          value: "production"
        - name: REACT_APP_API_URL
          value: "https://api.saske.ai"
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "500m"
            memory: "512Mi"
        readinessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 15
          periodSeconds: 20
---
apiVersion: v1
kind: Service
metadata:
  name: saske-frontend
  namespace: production
spec:
  type: LoadBalancer
  ports:
  - port: 443
    targetPort: 3000
    protocol: TCP
    name: https
  selector:
    app: saske-frontend
```

### CI/CD Pipeline

```yaml
# .github/workflows/production.yml
name: Production Deployment

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v1
      
    - name: Login to DockerHub
      uses: docker/login-action@v1
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
        
    - name: Build and push
      uses: docker/build-push-action@v2
      with:
        context: .
        push: true
        tags: saske-ai:production
        
    - name: Deploy to Kubernetes
      uses: azure/k8s-deploy@v1
      with:
        manifests: |
          k8s/production/*.yaml
        images: |
          saske-ai:production
        namespace: production
```

## Мониторинг развертывания

### Метрики

```typescript
// Конфигурация метрик
const deploymentMetrics = {
  duration: new Histogram({
    name: 'saske_deployment_duration_seconds',
    help: 'Duration of deployment in seconds'
  }),
  status: new Gauge({
    name: 'saske_deployment_status',
    help: 'Status of deployment (1: success, 0: failure)'
  }),
  pods: new Gauge({
    name: 'saske_deployment_pods',
    help: 'Number of pods in deployment'
  })
};
```

### Алерты

```yaml
# Правила алертов
groups:
- name: deployment
  rules:
  - alert: DeploymentFailed
    expr: saske_deployment_status == 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: Deployment failed
      description: Deployment process failed for 5 minutes

  - alert: PodsNotReady
    expr: saske_deployment_pods < 3
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: Not enough pods
      description: Number of ready pods is below 3
```

## Откат развертывания

### Kubernetes

```bash
# Скрипт отката
#!/bin/bash

# Получение предыдущей версии
PREVIOUS_VERSION=$(kubectl rollout history deployment/saske-frontend -n production | grep -A1 "REVISION" | tail -n1 | awk '{print $1}')

# Откат к предыдущей версии
kubectl rollout undo deployment/saske-frontend -n production --to-revision=$PREVIOUS_VERSION

# Ожидание завершения отката
kubectl rollout status deployment/saske-frontend -n production
```

### Docker

```bash
# Скрипт отката Docker
#!/bin/bash

# Получение предыдущего тега
PREVIOUS_TAG=$(docker images saske-ai --format "{{.Tag}}" | grep -v "latest" | sort -r | head -n1)

# Остановка текущих контейнеров
docker-compose down

# Запуск с предыдущей версией
docker-compose up -d --no-deps --build saske-ai:$PREVIOUS_TAG
```

## Лучшие практики

### Планирование

1. Определите стратегию развертывания
2. Выберите инструменты
3. Настройте окружения
4. Подготовьте документацию

### Выполнение

1. Используйте автоматизацию
2. Проводите тестирование
3. Мониторьте процесс
4. Документируйте изменения

### Мониторинг

1. Настройте алерты
2. Отслеживайте метрики
3. Анализируйте логи
4. Проверяйте здоровье

### Откат

1. Подготовьте процедуры
2. Протестируйте откат
3. Документируйте шаги
4. Тренируйте команду

## Ресурсы

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Helm Documentation](https://helm.sh/docs/) 