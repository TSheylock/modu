# Руководство по масштабированию

## Обзор

SASKE.ai использует многоуровневый подход к масштабированию, включающий:

1. Горизонтальное масштабирование
2. Вертикальное масштабирование
3. Масштабирование базы данных
4. Масштабирование кэша
5. Масштабирование очередей

## Горизонтальное масштабирование

### Kubernetes

```yaml
# Конфигурация деплоймента
apiVersion: apps/v1
kind: Deployment
metadata:
  name: saske-ai
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: saske-ai
  template:
    metadata:
      labels:
        app: saske-ai
    spec:
      containers:
      - name: saske-ai
        image: saske-ai:latest
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
```

### Load Balancer

```yaml
# Конфигурация сервиса
apiVersion: v1
kind: Service
metadata:
  name: saske-ai
  namespace: production
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 3000
  selector:
    app: saske-ai
```

## Вертикальное масштабирование

### Ресурсы

```yaml
# Конфигурация ресурсов
resources:
  requests:
    cpu: "1"
    memory: "1Gi"
  limits:
    cpu: "2"
    memory: "2Gi"
```

### Автомасштабирование

```yaml
# Конфигурация HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: saske-ai
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: saske-ai
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Масштабирование базы данных

### Репликация

```yaml
# Конфигурация MongoDB
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mongodb
  namespace: production
spec:
  serviceName: mongodb
  replicas: 3
  selector:
    matchLabels:
      app: mongodb
  template:
    metadata:
      labels:
        app: mongodb
    spec:
      containers:
      - name: mongodb
        image: mongo:latest
        args:
        - "--replSet"
        - "rs0"
        - "--bind_ip_all"
        ports:
        - containerPort: 27017
        volumeMounts:
        - name: data
          mountPath: /data/db
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 10Gi
```

### Шардирование

```javascript
// Конфигурация шардирования
const shardingConfig = {
  shards: [
    { host: 'shard1:27017' },
    { host: 'shard2:27017' },
    { host: 'shard3:27017' }
  ],
  databases: [
    {
      name: 'saske',
      collections: [
        {
          name: 'emotions',
          key: { userId: 'hashed' }
        }
      ]
    }
  ]
};
```

## Масштабирование кэша

### Redis Cluster

```yaml
# Конфигурация Redis Cluster
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: production
spec:
  serviceName: redis
  replicas: 6
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:latest
        args:
        - "--cluster-enabled"
        - "yes"
        - "--cluster-config-file"
        - "/data/nodes.conf"
        - "--cluster-node-timeout"
        - "5000"
        ports:
        - containerPort: 6379
        - containerPort: 16379
        volumeMounts:
        - name: data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 5Gi
```

## Масштабирование очередей

### RabbitMQ Cluster

```yaml
# Конфигурация RabbitMQ Cluster
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: rabbitmq
  namespace: production
spec:
  serviceName: rabbitmq
  replicas: 3
  selector:
    matchLabels:
      app: rabbitmq
  template:
    metadata:
      labels:
        app: rabbitmq
    spec:
      containers:
      - name: rabbitmq
        image: rabbitmq:3-management
        args:
        - "--cluster"
        - "--cluster-name"
        - "saske"
        ports:
        - containerPort: 5672
        - containerPort: 15672
        volumeMounts:
        - name: data
          mountPath: /var/lib/rabbitmq
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 5Gi
```

## Мониторинг масштабирования

### Метрики

```typescript
// Конфигурация метрик
const scalingMetrics = {
  cpu: new Gauge({
    name: 'saske_cpu_usage',
    help: 'CPU usage per pod'
  }),
  memory: new Gauge({
    name: 'saske_memory_usage',
    help: 'Memory usage per pod'
  }),
  requests: new Counter({
    name: 'saske_requests_total',
    help: 'Total number of requests'
  }),
  latency: new Histogram({
    name: 'saske_request_duration_seconds',
    help: 'Request duration in seconds'
  })
};
```

### Алерты

```yaml
# Правила алертов
groups:
- name: scaling
  rules:
  - alert: HighCPUUsage
    expr: avg(rate(saske_cpu_usage[5m])) > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High CPU usage detected
      description: CPU usage is above 80% for 5 minutes

  - alert: HighMemoryUsage
    expr: avg(rate(saske_memory_usage[5m])) > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High memory usage detected
      description: Memory usage is above 80% for 5 minutes
```

## Лучшие практики

### Горизонтальное масштабирование

1. Используйте без статуса приложения
2. Настраивайте балансировку нагрузки
3. Мониторьте производительность
4. Планируйте емкость

### Вертикальное масштабирование

1. Оптимизируйте ресурсы
2. Настраивайте автомасштабирование
3. Мониторьте использование
4. Планируйте рост

### База данных

1. Используйте репликацию
2. Настраивайте шардирование
3. Оптимизируйте запросы
4. Планируйте бэкапы

### Кэш

1. Используйте кластеризацию
2. Настраивайте TTL
3. Мониторьте hit ratio
4. Планируйте емкость

### Очереди

1. Используйте кластеризацию
2. Настраивайте очереди
3. Мониторьте latency
4. Планируйте обработку

## Ресурсы

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [Redis Documentation](https://redis.io/documentation)
- [RabbitMQ Documentation](https://www.rabbitmq.com/documentation.html) 