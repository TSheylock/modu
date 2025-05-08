# Emotion Diary с интеграцией блокчейна Saske

Приложение для анализа эмоций с использованием ИИ и блокчейна Saske для хранения и токенизации эмоциональных данных.

## Возможности

- Анализ эмоций с использованием различных модальностей (текст, аудио, видео)
- Визуализация эмоциональных данных в виде графов
- Создание NFT токенов эмоций в блокчейне Saske
- Хранение истории эмоций в децентрализованном хранилище
- Анализ трендов и паттернов эмоций

## Технологии

- React + TypeScript для фронтенда
- Solidity для смарт-контрактов
- Hardhat для разработки и деплоя
- IPFS для хранения данных
- Ethers.js и Web3.js для взаимодействия с блокчейном
- Chart.js и vis-network для визуализации
- NLP библиотеки для анализа текста

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/your-username/emotion-diary.git
cd emotion-diary
```

2. Установите зависимости:
```bash
npm install
```

3. Создайте файл .env на основе .env.example:
```bash
cp .env.example .env
```

4. Заполните необходимые переменные окружения в .env:
- PRIVATE_KEY - приватный ключ для подписи транзакций
- SASKE_MAINNET_RPC_URL - RPC URL для сети Saske Mainnet
- SASKE_TESTNET_RPC_URL - RPC URL для сети Saske Testnet
- IPFS_API_KEY - API ключ для IPFS
- ETHERSCAN_API_KEY - API ключ для верификации контрактов

## Разработка

1. Запустите локальный сервер разработки:
```bash
npm run dev
```

2. Скомпилируйте смарт-контракты:
```bash
npm run compile
```

3. Запустите тесты:
```bash
npm run test
```

## Деплой

### Testnet

1. Деплой смарт-контрактов в тестовую сеть:
```bash
npm run deploy:testnet
```

2. Верификация контрактов:
```bash
npm run verify:testnet
```

### Mainnet

1. Деплой смарт-контрактов в основную сеть:
```bash
npm run deploy:mainnet
```

2. Верификация контрактов:
```bash
npm run verify:mainnet
```

## Структура проекта

```
emotion-diary/
├── src/
│   ├── components/     # React компоненты
│   ├── contracts/      # Смарт-контракты
│   ├── utils/          # Утилиты
│   └── pages/          # Next.js страницы
├── scripts/            # Скрипты деплоя
├── test/               # Тесты
├── hardhat.config.ts   # Конфигурация Hardhat
└── package.json        # Зависимости и скрипты
```

## Лицензия

MIT 