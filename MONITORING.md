# Руководство по мониторингу

## Обзор

SASKE.ai использует комплексный подход к мониторингу, включающий:

1. Метрики приложения
2. Логирование
3. Трейсинг
4. Алерты
5. Дашборды
6. Интеграции

## Метрики приложения

### Prometheus

```yaml
# Конфигурация Prometheus
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'saske-ai'
    static_configs:
      - targets: ['localhost:3000']
    metrics_path: '/metrics'
    scheme: 'http'

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
```

### Метрики приложения

```typescript
// Конфигурация метрик
const metrics = {
  httpRequestDuration: new Histogram({
    name: 'saske_http_request_duration_seconds',
    help: 'Duration of HTTP requests in seconds',
    labelNames: ['method', 'route', 'status_code']
  }),
  
  activeUsers: new Gauge({
    name: 'saske_active_users',
    help: 'Number of active users'
  }),
  
  emotionAnalysisDuration: new Histogram({
    name: 'saske_emotion_analysis_duration_seconds',
    help: 'Duration of emotion analysis in seconds',
    labelNames: ['modality']
  }),
  
  blockchainTransactions: new Counter({
    name: 'saske_blockchain_transactions_total',
    help: 'Total number of blockchain transactions',
    labelNames: ['type', 'status']
  })
};
```

## Логирование

### ELK Stack

```yaml
# Конфигурация Elasticsearch
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.17.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:7.17.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
    ports:
      - "5044:5044"
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:7.17.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

### Структура логов

```typescript
// Конфигурация Winston
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  defaultMeta: { service: 'saske-ai' },
  transports: [
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' })
  ]
});

// Пример структуры лога
logger.info('User action', {
  userId: 'user123',
  action: 'emotion_analysis',
  modality: 'text',
  duration: 0.5,
  result: 'happy',
  confidence: 0.92
});
```

## Трейсинг

### Jaeger

```yaml
# Конфигурация Jaeger
version: '3.8'
services:
  jaeger:
    image: jaegertracing/all-in-one:1.30
    ports:
      - "5775:5775/udp"
      - "6831:6831/udp"
      - "6832:6832/udp"
      - "5778:5778"
      - "16686:16686"
      - "14250:14250"
      - "14268:14268"
      - "14269:14269"
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
```

### Инструментация

```typescript
// Конфигурация OpenTelemetry
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { Resource } from '@opentelemetry/resources';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';
import { JaegerExporter } from '@opentelemetry/exporter-jaeger';

const provider = new NodeTracerProvider({
  resource: new Resource({
    [SemanticResourceAttributes.SERVICE_NAME]: 'saske-ai',
  }),
});

const exporter = new JaegerExporter({
  endpoint: 'http://localhost:14268/api/traces',
});

provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
provider.register();

// Пример трейсинга
const tracer = trace.getTracer('saske-ai');

const analyzeEmotion = async (text: string) => {
  const span = tracer.startSpan('analyze_emotion');
  try {
    span.setAttribute('text_length', text.length);
    // Логика анализа эмоций
    return result;
  } finally {
    span.end();
  }
};
```

## Алерты

### Alertmanager

```yaml
# Конфигурация Alertmanager
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/YOUR_SLACK_WEBHOOK'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'slack-notifications'

receivers:
- name: 'slack-notifications'
  slack_configs:
  - channel: '#alerts'
    send_resolved: true
    title: '{{ template "slack.default.title" . }}'
    text: '{{ template "slack.default.text" . }}'
```

### Правила алертов

```yaml
# Правила алертов
groups:
- name: saske-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(saske_http_requests_total{status_code=~"5.."}[5m]) / rate(saske_http_requests_total[5m]) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: High error rate detected
      description: Error rate is above 5% for 5 minutes

  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(saske_http_request_duration_seconds_bucket[5m])) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High latency detected
      description: 95th percentile latency is above 1s

  - alert: LowConfidenceEmotionAnalysis
    expr: histogram_quantile(0.5, rate(saske_emotion_analysis_confidence_bucket[5m])) < 0.7
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: Low confidence in emotion analysis
      description: Median confidence in emotion analysis is below 70%
```

## Дашборды

### Grafana

```json
{
  "dashboard": {
    "id": null,
    "title": "SASKE.ai Overview",
    "tags": ["saske", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "title": "HTTP Request Rate",
        "type": "graph",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "rate(saske_http_requests_total[5m])",
            "legendFormat": "{{method}} {{route}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "rate(saske_http_requests_total{status_code=~\"5..\"}[5m]) / rate(saske_http_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      },
      {
        "title": "Active Users",
        "type": "gauge",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "saske_active_users",
            "legendFormat": "Active Users"
          }
        ]
      }
    ]
  }
}
```

### Дашборд анализа эмоций

```json
{
  "dashboard": {
    "id": null,
    "title": "SASKE.ai Emotion Analysis",
    "tags": ["saske", "emotions"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Emotion Distribution",
        "type": "piechart",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "sum(saske_emotion_results_total) by (emotion)",
            "legendFormat": "{{emotion}}"
          }
        ]
      },
      {
        "title": "Analysis Duration",
        "type": "graph",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "rate(saske_emotion_analysis_duration_seconds_sum[5m]) / rate(saske_emotion_analysis_duration_seconds_count[5m])",
            "legendFormat": "{{modality}}"
          }
        ]
      },
      {
        "title": "Confidence Level",
        "type": "gauge",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(saske_emotion_analysis_confidence_bucket[5m]))",
            "legendFormat": "95th Percentile Confidence"
          }
        ]
      }
    ]
  }
}
```

## Интеграции

### Slack

```typescript
// Конфигурация Slack
const slackConfig = {
  webhookUrl: process.env.SLACK_WEBHOOK_URL,
  channel: '#saske-alerts',
  username: 'SASKE.ai Monitor',
  iconEmoji: ':robot_face:'
};

// Отправка уведомления в Slack
const sendSlackNotification = async (message: string, level: 'info' | 'warning' | 'error') => {
  const color = level === 'error' ? '#FF0000' : level === 'warning' ? '#FFA500' : '#00FF00';
  
  await axios.post(slackConfig.webhookUrl, {
    channel: slackConfig.channel,
    username: slackConfig.username,
    icon_emoji: slackConfig.iconEmoji,
    attachments: [
      {
        color,
        text: message,
        fields: [
          {
            title: 'Environment',
            value: process.env.NODE_ENV,
            short: true
          },
          {
            title: 'Timestamp',
            value: new Date().toISOString(),
            short: true
          }
        ]
      }
    ]
  });
};
```

### Email

```typescript
// Конфигурация Email
const emailConfig = {
  host: process.env.SMTP_HOST,
  port: process.env.SMTP_PORT,
  secure: true,
  auth: {
    user: process.env.SMTP_USER,
    pass: process.env.SMTP_PASS
  }
};

// Отправка уведомления по email
const sendEmailNotification = async (subject: string, message: string) => {
  const transporter = nodemailer.createTransport(emailConfig);
  
  await transporter.sendMail({
    from: `"SASKE.ai Monitor" <${emailConfig.auth.user}>`,
    to: process.env.ALERT_EMAIL,
    subject: `[SASKE.ai Alert] ${subject}`,
    html: `
      <h2>SASKE.ai Alert</h2>
      <p><strong>Environment:</strong> ${process.env.NODE_ENV}</p>
      <p><strong>Timestamp:</strong> ${new Date().toISOString()}</p>
      <p><strong>Message:</strong></p>
      <p>${message}</p>
    `
  });
};
```

### Telegram

```typescript
// Конфигурация Telegram
const telegramConfig = {
  botToken: process.env.TELEGRAM_BOT_TOKEN,
  chatId: process.env.TELEGRAM_CHAT_ID
};

// Отправка уведомления в Telegram
const sendTelegramNotification = async (message: string) => {
  await axios.post(`https://api.telegram.org/bot${telegramConfig.botToken}/sendMessage`, {
    chat_id: telegramConfig.chatId,
    text: `🚨 *SASKE.ai Alert*\n\n${message}`,
    parse_mode: 'Markdown'
  });
};
```

## Лучшие практики

### Метрики

1. Используйте осмысленные имена
2. Добавляйте описания и единицы измерения
3. Группируйте связанные метрики
4. Используйте правильные типы метрик

### Логи

1. Используйте структурированное логирование
2. Добавляйте контекст к логам
3. Настраивайте уровни логирования
4. Регулярно ротируйте логи

### Трейсинг

1. Трейсьте критические операции
2. Добавляйте атрибуты к спанам
3. Используйте правильную вложенность
4. Анализируйте трейсы

### Алерты

1. Настраивайте разумные пороги
2. Группируйте связанные алерты
3. Добавляйте описания и действия
4. Тестируйте алерты

### Дашборды

1. Создавайте логичные группировки
2. Используйте подходящие визуализации
3. Добавляйте аннотации
4. Настраивайте автообновление

## Ресурсы

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [ELK Stack Documentation](https://www.elastic.co/guide/index.html)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/) 