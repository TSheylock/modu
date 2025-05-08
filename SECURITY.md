# Руководство по безопасности

## Обзор

SASKE.ai использует многоуровневый подход к безопасности, включающий:

1. Аутентификацию и авторизацию
2. Шифрование данных
3. Защиту API
4. Мониторинг безопасности
5. Управление уязвимостями

## Аутентификация и авторизация

### JWT

```typescript
// Конфигурация JWT
import jwt from 'jsonwebtoken';

const jwtConfig = {
  secret: process.env.JWT_SECRET,
  expiresIn: '1h',
  algorithm: 'HS256'
};

// Создание токена
const createToken = (payload: any): string => {
  return jwt.sign(payload, jwtConfig.secret, {
    expiresIn: jwtConfig.expiresIn,
    algorithm: jwtConfig.algorithm
  });
};

// Верификация токена
const verifyToken = (token: string): any => {
  try {
    return jwt.verify(token, jwtConfig.secret);
  } catch (error) {
    throw new Error('Invalid token');
  }
};

// Middleware для проверки токена
const authMiddleware = async (req: Request, res: Response, next: NextFunction) => {
  const token = req.headers.authorization?.split(' ')[1];
  
  if (!token) {
    return res.status(401).json({ error: 'No token provided' });
  }
  
  try {
    const decoded = verifyToken(token);
    req.user = decoded;
    next();
  } catch (error) {
    return res.status(401).json({ error: 'Invalid token' });
  }
};
```

### OAuth 2.0

```typescript
// Конфигурация OAuth 2.0
import { OAuth2Client } from 'google-auth-library';

const oauth2Client = new OAuth2Client(
  process.env.GOOGLE_CLIENT_ID,
  process.env.GOOGLE_CLIENT_SECRET,
  process.env.GOOGLE_REDIRECT_URI
);

// Аутентификация через Google
const googleAuth = async (code: string) => {
  const { tokens } = await oauth2Client.getToken(code);
  oauth2Client.setCredentials(tokens);
  
  const ticket = await oauth2Client.verifyIdToken({
    idToken: tokens.id_token,
    audience: process.env.GOOGLE_CLIENT_ID
  });
  
  const payload = ticket.getPayload();
  return {
    id: payload.sub,
    email: payload.email,
    name: payload.name,
    picture: payload.picture
  };
};
```

## Шифрование данных

### Шифрование в покое

```typescript
// Конфигурация шифрования
import crypto from 'crypto';

const encryptionConfig = {
  algorithm: 'aes-256-gcm',
  key: Buffer.from(process.env.ENCRYPTION_KEY, 'hex'),
  ivLength: 12,
  saltLength: 16,
  tagLength: 16
};

// Шифрование данных
const encryptData = (data: string): string => {
  const iv = crypto.randomBytes(encryptionConfig.ivLength);
  const salt = crypto.randomBytes(encryptionConfig.saltLength);
  
  const cipher = crypto.createCipheriv(
    encryptionConfig.algorithm,
    encryptionConfig.key,
    iv
  );
  
  const encrypted = Buffer.concat([
    cipher.update(data, 'utf8'),
    cipher.final()
  ]);
  
  const tag = cipher.getAuthTag();
  
  return Buffer.concat([salt, iv, tag, encrypted]).toString('base64');
};

// Дешифрование данных
const decryptData = (encryptedData: string): string => {
  const buffer = Buffer.from(encryptedData, 'base64');
  
  const salt = buffer.slice(0, encryptionConfig.saltLength);
  const iv = buffer.slice(
    encryptionConfig.saltLength,
    encryptionConfig.saltLength + encryptionConfig.ivLength
  );
  const tag = buffer.slice(
    encryptionConfig.saltLength + encryptionConfig.ivLength,
    encryptionConfig.saltLength + encryptionConfig.ivLength + encryptionConfig.tagLength
  );
  const encrypted = buffer.slice(
    encryptionConfig.saltLength + encryptionConfig.ivLength + encryptionConfig.tagLength
  );
  
  const decipher = crypto.createDecipheriv(
    encryptionConfig.algorithm,
    encryptionConfig.key,
    iv
  );
  
  decipher.setAuthTag(tag);
  
  return Buffer.concat([
    decipher.update(encrypted),
    decipher.final()
  ]).toString('utf8');
};
```

### Шифрование в пути

```typescript
// Конфигурация HTTPS
import https from 'https';
import fs from 'fs';

const httpsOptions = {
  key: fs.readFileSync(process.env.SSL_KEY_PATH),
  cert: fs.readFileSync(process.env.SSL_CERT_PATH),
  ca: fs.readFileSync(process.env.SSL_CA_PATH),
  requestCert: true,
  rejectUnauthorized: true
};

// Создание HTTPS сервера
const server = https.createServer(httpsOptions, app);

// Настройка заголовков безопасности
app.use((req, res, next) => {
  res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  next();
});
```

## Защита API

### Ограничение скорости

```typescript
// Конфигурация rate limiting
import rateLimit from 'express-rate-limit';

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 минут
  max: 100, // максимум 100 запросов
  standardHeaders: true,
  legacyHeaders: false,
  message: 'Too many requests, please try again later'
});

// Применение rate limiting
app.use('/api/', limiter);
```

### CORS

```typescript
// Конфигурация CORS
import cors from 'cors';

const corsOptions = {
  origin: process.env.CORS_ORIGIN,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  exposedHeaders: ['Content-Range', 'X-Content-Range'],
  credentials: true,
  maxAge: 86400 // 24 часа
};

// Применение CORS
app.use(cors(corsOptions));
```

### Helmet

```typescript
// Конфигурация Helmet
import helmet from 'helmet';

// Применение Helmet
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", 'data:', 'https:'],
      connectSrc: ["'self'", 'https://api.saske.ai'],
      fontSrc: ["'self'"],
      objectSrc: ["'none'"],
      mediaSrc: ["'self'"],
      frameSrc: ["'none'"]
    }
  },
  crossOriginEmbedderPolicy: true,
  crossOriginOpenerPolicy: true,
  crossOriginResourcePolicy: { policy: 'same-site' },
  dnsPrefetchControl: true,
  frameguard: { action: 'deny' },
  hidePoweredBy: true,
  hsts: true,
  ieNoOpen: true,
  noSniff: true,
  referrerPolicy: { policy: 'strict-origin-when-cross-origin' },
  xssFilter: true
}));
```

## Мониторинг безопасности

### Аудит логирования

```typescript
// Конфигурация аудит логирования
import winston from 'winston';

const auditLogger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  defaultMeta: { service: 'saske-ai-audit' },
  transports: [
    new winston.transports.File({ filename: 'logs/audit.log' })
  ]
});

// Пример аудит лога
const logAudit = (userId: string, action: string, resource: string, status: string) => {
  auditLogger.info('Security audit', {
    userId,
    action,
    resource,
    status,
    timestamp: new Date().toISOString(),
    ip: req.ip,
    userAgent: req.headers['user-agent']
  });
};
```

### Сканирование уязвимостей

```yaml
# Конфигурация OWASP ZAP
version: '3.8'
services:
  zap:
    image: owasp/zap2docker-stable
    ports:
      - "8080:8080"
    volumes:
      - ./zap:/zap/wrk
    command: zap-baseline.py -t http://saske-ai:3000 -g gen.conf -r zap-report.html
```

## Управление уязвимостями

### Процесс управления уязвимостями

1. Обнаружение уязвимостей
2. Оценка рисков
3. Приоритизация исправлений
4. Разработка патчей
5. Тестирование патчей
6. Развертывание исправлений
7. Верификация исправлений

### Шаблон отчета об уязвимости

```markdown
# Отчет об уязвимости

## Общая информация
- **ID уязвимости**: VULN-2023-001
- **Дата обнаружения**: 2023-01-15
- **Статус**: Открыто
- **Приоритет**: Высокий

## Описание
Краткое описание уязвимости и ее потенциального воздействия.

## Технические детали
- **Тип уязвимости**: SQL Injection
- **Компонент**: API аутентификации
- **Версия**: 1.2.3
- **CVE**: CVE-2023-12345

## Шаги воспроизведения
1. Шаг 1
2. Шаг 2
3. Шаг 3

## Риск
- **CVSS**: 8.5 (Высокий)
- **Влияние**: Неавторизованный доступ к данным пользователей

## Рекомендации по исправлению
1. Использовать параметризованные запросы
2. Добавить валидацию входных данных
3. Обновить библиотеку до последней версии

## План исправления
- **Ответственный**: Команда безопасности
- **Срок**: 2023-01-22
- **Метод развертывания**: Горячее исправление
```

## Лучшие практики

### Разработка

1. Используйте безопасные библиотеки
2. Следуйте принципу наименьших привилегий
3. Валидируйте все входные данные
4. Используйте параметризованные запросы
5. Регулярно обновляйте зависимости

### Операции

1. Используйте HTTPS везде
2. Настройте файрволы
3. Регулярно проводите сканирование уязвимостей
4. Мониторьте подозрительную активность
5. Создайте план реагирования на инциденты

### Данные

1. Шифруйте чувствительные данные
2. Используйте безопасное хранение паролей
3. Регулярно делайте резервные копии
4. Ограничивайте доступ к данным
5. Следуйте принципу наименьших привилегий

## Ресурсы

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [ISO 27001](https://www.iso.org/isoiec-27001-information-security.html)
- [CIS Benchmarks](https://www.cisecurity.org/cis-benchmarks/) 