# Руководство по оптимизации

## Обзор

SASKE.ai использует комплексный подход к оптимизации, включающий:

1. Оптимизацию производительности
2. Оптимизацию кода
3. Оптимизацию базы данных
4. Оптимизацию сети
5. Оптимизацию ресурсов

## Оптимизация производительности

### Профилирование

```typescript
// Конфигурация профайлера
import { Profiler } from '@opentelemetry/api';

const profiler = new Profiler({
  serviceName: 'saske-ai',
  environment: process.env.NODE_ENV,
  samplingRate: 0.1, // 10% запросов
  maxSamples: 1000
});

// Профилирование функции
const profileFunction = async (fn: Function, name: string) => {
  const span = profiler.startSpan(name);
  try {
    return await fn();
  } finally {
    span.end();
  }
};

// Пример использования
const analyzeEmotion = async (text: string) => {
  return profileFunction(async () => {
    // Логика анализа эмоций
    return result;
  }, 'analyze_emotion');
};
```

### Кэширование

```typescript
// Конфигурация Redis
import Redis from 'ioredis';

const redis = new Redis({
  host: process.env.REDIS_HOST,
  port: parseInt(process.env.REDIS_PORT),
  password: process.env.REDIS_PASSWORD,
  db: 0,
  keyPrefix: 'saske:'
});

// Кэширование результатов
const cacheResult = async (key: string, data: any, ttl: number = 3600) => {
  await redis.set(key, JSON.stringify(data), 'EX', ttl);
  return data;
};

// Получение из кэша
const getCachedResult = async (key: string) => {
  const cached = await redis.get(key);
  return cached ? JSON.parse(cached) : null;
};

// Пример использования
const getEmotionAnalysis = async (text: string) => {
  const cacheKey = `emotion:${text}`;
  const cached = await getCachedResult(cacheKey);
  
  if (cached) {
    return cached;
  }
  
  const result = await analyzeEmotion(text);
  return cacheResult(cacheKey, result);
};
```

## Оптимизация кода

### Линтинг

```json
// Конфигурация ESLint
{
  "extends": [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:react/recommended",
    "plugin:react-hooks/recommended",
    "plugin:prettier/recommended"
  ],
  "plugins": [
    "@typescript-eslint",
    "react",
    "react-hooks",
    "prettier"
  ],
  "rules": {
    "no-unused-vars": "error",
    "no-console": "warn",
    "prettier/prettier": "error",
    "react/prop-types": "off",
    "@typescript-eslint/explicit-function-return-type": "warn",
    "@typescript-eslint/no-explicit-any": "warn"
  },
  "settings": {
    "react": {
      "version": "detect"
    }
  }
}
```

### Форматирование

```json
// Конфигурация Prettier
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 100,
  "tabWidth": 2,
  "useTabs": false,
  "bracketSpacing": true,
  "arrowParens": "avoid",
  "endOfLine": "lf"
}
```

## Оптимизация базы данных

### Индексация

```typescript
// Схема с индексами
import { Schema, model } from 'mongoose';

const userSchema = new Schema({
  email: {
    type: String,
    required: true,
    unique: true,
    index: true
  },
  username: {
    type: String,
    required: true,
    unique: true,
    index: true
  },
  createdAt: {
    type: Date,
    default: Date.now,
    index: true
  },
  lastLogin: {
    type: Date,
    index: true
  }
});

// Составной индекс
userSchema.index({ email: 1, createdAt: -1 });

// Текстовый индекс для поиска
userSchema.index({ username: 'text', email: 'text' });

const User = model('User', userSchema);
```

### Оптимизированные запросы

```typescript
// Оптимизированный запрос
const getActiveUsers = async (days: number = 30) => {
  const cutoffDate = new Date();
  cutoffDate.setDate(cutoffDate.getDate() - days);
  
  return User.find({
    lastLogin: { $gte: cutoffDate }
  })
  .select('email username lastLogin')
  .sort({ lastLogin: -1 })
  .limit(100)
  .lean()
  .exec();
};

// Агрегация для аналитики
const getUserStats = async () => {
  return User.aggregate([
    {
      $group: {
        _id: {
          year: { $year: '$createdAt' },
          month: { $month: '$createdAt' }
        },
        count: { $sum: 1 },
        activeUsers: {
          $sum: {
            $cond: [
              { $gte: ['$lastLogin', new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)] },
              1,
              0
            ]
          }
        }
      }
    },
    {
      $sort: { '_id.year': -1, '_id.month': -1 }
    }
  ]);
};
```

## Оптимизация сети

### Сжатие

```typescript
// Конфигурация сжатия
import compression from 'compression';

// Применение сжатия
app.use(compression({
  level: 6, // уровень сжатия (0-9)
  threshold: 1024, // минимальный размер для сжатия (в байтах)
  filter: (req, res) => {
    if (req.headers['x-no-compression']) {
      return false;
    }
    return compression.filter(req, res);
  }
}));
```

### HTTP кэширование

```typescript
// Конфигурация кэширования
import { cacheControl } from 'express-cache-controller';

// Применение кэширования
app.use(cacheControl({
  maxAge: 3600, // время жизни кэша в секундах
  sMaxAge: 86400, // время жизни кэша на прокси в секундах
  public: true, // публичный кэш
  private: false, // приватный кэш
  noCache: false, // запрет кэширования
  noStore: false, // запрет хранения
  mustRevalidate: true, // обязательная ревалидация
  staleWhileRevalidate: 60 // время использования устаревшего кэша при ревалидации
}));

// Пример использования для конкретного маршрута
app.get('/api/emotions', (req, res) => {
  res.cacheControl({
    maxAge: 300, // 5 минут
    public: true
  });
  
  // Логика получения данных
  res.json(data);
});
```

## Оптимизация ресурсов

### Оптимизация изображений

```typescript
// Конфигурация Sharp
import sharp from 'sharp';

// Оптимизация изображения
const optimizeImage = async (input: Buffer, options: {
  width?: number;
  height?: number;
  quality?: number;
  format?: 'jpeg' | 'png' | 'webp';
}) => {
  let image = sharp(input);
  
  if (options.width || options.height) {
    image = image.resize(options.width, options.height, {
      fit: 'inside',
      withoutEnlargement: true
    });
  }
  
  switch (options.format) {
    case 'jpeg':
      image = image.jpeg({ quality: options.quality || 80 });
      break;
    case 'png':
      image = image.png({ quality: options.quality || 80 });
      break;
    case 'webp':
      image = image.webp({ quality: options.quality || 80 });
      break;
    default:
      image = image.jpeg({ quality: options.quality || 80 });
  }
  
  return image.toBuffer();
};

// Пример использования
const processUserAvatar = async (file: Express.Multer.File) => {
  const optimized = await optimizeImage(file.buffer, {
    width: 200,
    height: 200,
    quality: 80,
    format: 'webp'
  });
  
  return optimized;
};
```

### Оптимизация бандла

```javascript
// Конфигурация Webpack
const path = require('path');
const TerserPlugin = require('terser-webpack-plugin');
const CompressionPlugin = require('compression-webpack-plugin');
const { BundleAnalyzerPlugin } = require('webpack-bundle-analyzer');

module.exports = {
  mode: 'production',
  entry: './src/index.tsx',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].[contenthash].js',
    chunkFilename: '[name].[contenthash].chunk.js',
    publicPath: '/'
  },
  optimization: {
    minimize: true,
    minimizer: [
      new TerserPlugin({
        terserOptions: {
          compress: {
            drop_console: true,
            drop_debugger: true
          }
        }
      })
    ],
    splitChunks: {
      chunks: 'all',
      maxInitialRequests: Infinity,
      minSize: 0,
      cacheGroups: {
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name(module) {
            const packageName = module.context.match(/[\\/]node_modules[\\/](.*?)([\\/]|$)/)[1];
            return `vendor.${packageName.replace('@', '')}`;
          }
        }
      }
    }
  },
  plugins: [
    new CompressionPlugin({
      algorithm: 'gzip',
      test: /\.(js|css|html|svg)$/,
      threshold: 10240,
      minRatio: 0.8
    }),
    new BundleAnalyzerPlugin({
      analyzerMode: process.env.ANALYZE === 'true' ? 'server' : 'disabled'
    })
  ]
};
```

## Мониторинг производительности

### Метрики

```typescript
// Конфигурация метрик
import { Histogram, Gauge, Counter } from 'prom-client';

const performanceMetrics = {
  responseTime: new Histogram({
    name: 'saske_http_response_time_seconds',
    help: 'HTTP response time in seconds',
    labelNames: ['method', 'route', 'status_code']
  }),
  
  memoryUsage: new Gauge({
    name: 'saske_memory_usage_bytes',
    help: 'Memory usage in bytes',
    labelNames: ['type']
  }),
  
  activeConnections: new Gauge({
    name: 'saske_active_connections',
    help: 'Number of active connections'
  }),
  
  errors: new Counter({
    name: 'saske_errors_total',
    help: 'Total number of errors',
    labelNames: ['type', 'code']
  })
};

// Обновление метрик
const updateMetrics = () => {
  const memory = process.memoryUsage();
  
  performanceMetrics.memoryUsage.set({ type: 'heapUsed' }, memory.heapUsed);
  performanceMetrics.memoryUsage.set({ type: 'heapTotal' }, memory.heapTotal);
  performanceMetrics.memoryUsage.set({ type: 'rss' }, memory.rss);
  
  performanceMetrics.activeConnections.set(server.connections);
};
```

### Алерты

```yaml
# Правила алертов для производительности
groups:
- name: performance
  rules:
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(saske_http_response_time_seconds_bucket[5m])) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High response time
      description: 95th percentile response time is above 1s

  - alert: HighMemoryUsage
    expr: saske_memory_usage_bytes{type="heapUsed"} / saske_memory_usage_bytes{type="heapTotal"} > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High memory usage
      description: Memory usage is above 90%

  - alert: HighErrorRate
    expr: rate(saske_errors_total[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: High error rate
      description: Error rate is above 0.1 per second
```

## Лучшие практики

### Код

1. Используйте профилирование для выявления узких мест
2. Оптимизируйте критические пути
3. Используйте кэширование для часто запрашиваемых данных
4. Минимизируйте количество запросов к базе данных
5. Используйте ленивую загрузку для больших ресурсов

### База данных

1. Создавайте правильные индексы
2. Оптимизируйте запросы
3. Используйте агрегацию для сложных запросов
4. Настройте пул соединений
5. Регулярно анализируйте производительность

### Сеть

1. Используйте сжатие для больших ответов
2. Настройте правильные заголовки кэширования
3. Минимизируйте размер ответов
4. Используйте CDN для статических ресурсов
5. Оптимизируйте порядок загрузки ресурсов

### Ресурсы

1. Оптимизируйте изображения
2. Минимизируйте и объединяйте CSS и JavaScript
3. Используйте современные форматы (WebP, AVIF)
4. Настройте правильное кэширование
5. Используйте ленивую загрузку для изображений

## Ресурсы

- [Web Performance Optimization](https://web.dev/fast/)
- [Node.js Performance](https://nodejs.org/en/docs/guides/performance/)
- [MongoDB Performance](https://www.mongodb.com/docs/manual/core/query-optimization/)
- [React Performance](https://reactjs.org/docs/optimizing-performance.html)
- [Webpack Optimization](https://webpack.js.org/guides/build-performance/) 