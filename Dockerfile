# Этап сборки
FROM node:18-alpine as builder

WORKDIR /app

# Копируем файлы зависимостей
COPY package*.json ./
COPY yarn.lock ./

# Устанавливаем зависимости
RUN yarn install --frozen-lockfile

# Копируем исходный код
COPY . .

# Собираем приложение
RUN yarn build

# Этап production
FROM node:18-alpine

WORKDIR /app

# Копируем собранное приложение
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/yarn.lock ./

# Устанавливаем только production зависимости
RUN yarn install --production --frozen-lockfile

# Открываем порт
EXPOSE 3000

# Запускаем приложение
CMD ["yarn", "start"] 