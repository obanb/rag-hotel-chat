import {Server} from "node:http";
import express from 'express';
import {ragChat} from "./chat";
import fs from "fs";

const port = process.env.PORT;

const app = express();

export const startServer = async () => {
    await new Promise<Server>((resolve) => {
        const httpServer = app.listen(port, () => {
            console.log(`[server]: Server is running at http://localhost:${port}`);
            resolve(httpServer);
        });
    });

    const chat = await ragChat();

    app.use(express.json({ limit: '5mb' }));

    app.get('/json', (req, res) => {
        const json = fs.readFileSync(`__SOURCES__/sources.json`, {encoding: 'utf-8'});
        res.send(JSON.stringify(JSON.parse(json), null, 2));
    })

    process.stdin.addListener('data', async function (input) {
        let userInput = input.toString().trim();
        await chat.chat(userInput);
    });
};

startServer().catch(console.error);
