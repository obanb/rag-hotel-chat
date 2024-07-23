##  RAG chat example setup

> Node.js RAG chat example for hotel guests

- **RAG:** Retrieval-Augmented Generation) - prompt > get context (Pinecone) > add context to query > ask OpenAI https://cloud.google.com/use-cases/retrieval-augmented-generation
- **Pinecone:** vector database for similarity search https://docs.pinecone.io/guides/get-started/quickstart
- **OpenAI Function calling**: bypasses context query for live data or performs an arbitrary action https://platform.openai.com/docs/guides/function-calling


### Installation

Package install:

```
$ yarn install
```

.env variable if you want to use expressions with OpenAI language models

```
OPEN_AI_KEY=xxx
```

.env variable for Pinecone API key

```
PINECONE_KEY=xxx
```

.env variable for Cosmos DB

```
COSMOS_URI=xxx
```

.env variable for database name

```
COSMOS_DB=xxx
```

.env variable for the server port

```
PORT=8080
```

start

```
$ yarn run start
```

### Usage

- the first run is to create a Pinecone index and populate it with data from the ```__SOURCE__``` folder
- just chat with terminal


