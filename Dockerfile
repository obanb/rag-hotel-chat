FROM node:18.19.0-alpine3.18

WORKDIR /usr/src/app

COPY . /usr/src/app

RUN yarn install --production=true

EXPOSE 8080

WORKDIR packages/api

CMD node lib/api.js