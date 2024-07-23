import OpenAI from "openai";
import {OpenAIEmbeddings} from "@langchain/openai";
import {Pinecone} from "@pinecone-database/pinecone";
import fs from "fs";
import {Document} from "@langchain/core/documents";
import {PineconeStore} from "@langchain/pinecone";
import {Collection, MongoClient} from "mongodb";


// RAG (Retrieval-Augmented Generation) chatbot example with OpenAI (and OpenAI Function calling) and Pinecone vector search
// RAG - query > get relevant context (Pinecone vector db) > update context > ask OpenAI (and maybe use Function calling before)

const COSMOS_URI = process.env.COSMOS_URI;
const COSMOS_DB = process.env.COSMOS_DB

const __SOURCE_DIR__ = './__SOURCES__'
const __PINECONE_IDX__ = 'rag-chat-hotels-idx'
//gpt-4o-mini - cheaper, more capable, just as fast version of  GPT-3.5 Turbo, much larger ctx window 16.385 vs 128.000 tokens
const __GPT_MODEL__ = 'gpt-4o-mini'
const __EMBEDDINGS__ = 'text-embedding-3-small'
// 1536 is the default dimension for OpenAI embeddings which im using
const __DIMENSIONS__ = 1536
const __PINECONE_CLOUD__ = 'aws'
const __PINECONE_REGION__ = 'us-east-1'

type SourceDocument<T> = {
    data: string;
    segment: string;
    subsegment: string;
    metadata: T;
}

type HotelSourceDocument = SourceDocument<{
    hotelName: string;
}>

export const ragChat = async() => {
    const cosmosClient = new MongoClient(COSMOS_URI!);
    await cosmosClient.connect()
    const db = cosmosClient.db(COSMOS_DB);
    const bokingColl = db.collection('bookingStatus');

    // OpenAI connect
    const openaiClient = new OpenAI({
        apiKey: process.env.OPENAI_API_KEY!
    });

    // newer & cheaper successor to text-embedding-ada-002, 1536 dimensions
    const embeddings = new OpenAIEmbeddings({
        model: __EMBEDDINGS__
    })
    // instance level context for demonstration
    const ctx: OpenAI.Chat.Completions.ChatCompletionMessageParam[]  = []

    const pineconeClient = new Pinecone({
        apiKey: process.env.PINECONE_KEY!,
    });

    const indexes = await pineconeClient!.listIndexes();
    const existingIndex = indexes.indexes?.map((index) => index.name).includes(__PINECONE_IDX__);
    if(!existingIndex) {
        console.log(`Creating Pinecone Index ${__PINECONE_IDX__}..`)
        await pineconeClient!.createIndex({
            name: __PINECONE_IDX__,
            dimension: __DIMENSIONS__,
            //  cosine because context and relative importance of features are more critical than their absolute values
            metric: 'cosine',
            spec: {
                serverless: {
                    cloud:  __PINECONE_CLOUD__,
                    region: __PINECONE_REGION__
                }
            }
        })

        const fileContent = fs.readFileSync(`${__SOURCE_DIR__}/sources.json`, {encoding: 'utf-8'});

        const sourceData: HotelSourceDocument[] = JSON.parse(fileContent);

        const docs: Document[] = sourceData.map((sourceDoc) => ({
            metadata: sourceDoc.metadata,
            pageContent: `[${sourceDoc.segment}][${sourceDoc.subsegment}] - ${JSON.stringify(sourceDoc.data)}`
        }));

        const pineconeIndex = pineconeClient.Index(__PINECONE_IDX__);

        await PineconeStore.fromDocuments(docs, embeddings, {
            pineconeIndex,
            maxConcurrency: 1,
        });
    }

    const pineconeIndex = pineconeClient.Index(__PINECONE_IDX__);

    const vectorStore = await PineconeStore.fromExistingIndex(
        embeddings,
        { pineconeIndex }
    );


    return {
        chat: async (prompt: string) => {
            ctx.push({
                role: 'system',
                content: 'You are a helpful chatbot communicating hotel information (rooms, reservations, hotel surroundings...'
            })

            let relevantMetadata: string | null = null
            const results = await vectorStore.similaritySearch(prompt, 1, {
                // there should be for example filter by hotel
            });

            if(results.length > 0) {
                ctx.push({
                    role: 'assistant',
                    content: `Answer the next question using following info: ${results[0].pageContent}`
                })
                relevantMetadata = JSON.stringify(`relevantInfo: ${JSON.stringify(results[0].metadata)}`);
            }
                ctx.push({
                    role: 'user',
                    content: prompt
                })

            const firstCall = await openaiClient.chat.completions.create({
                model: __GPT_MODEL__,
                messages: ctx,
                tools: [{
                    type: 'function',
                    function: {
                        name: 'getDateTime',
                        description: 'Get the current date and time'
                    }
                },
                    {
                        type: 'function',
                        function: {
                            name: 'getBookingStatus',
                            description: 'Get booking status/informations',
                            parameters: {
                                type: 'object',
                                properties: {
                                    bookingId: {
                                        type: 'string',
                                        description: 'booking id'
                                    }
                                },
                                required: ['bookingId']
                            }
                        }
                    },
                    {
                        type: 'function',
                        function: {
                            name: 'sendEmailToReception',
                            description: 'Send email to reception',
                            parameters: {
                                type: 'object',
                                properties: {
                                    content: {
                                        type: 'string',
                                        description: 'email content'
                                    }
                                },
                                required: ['bookingId']
                            }
                        }
                    }],
                // lets the model decide whether to call functions and, if so, which functions to call
                tool_choice: 'auto'
            }, )


            const invokeToolCalling = firstCall.choices[0].finish_reason === 'tool_calls'

            if(invokeToolCalling){
                const toolFn =  firstCall.choices[0].message.tool_calls![0]
                console.log(`calling tool_call fn: ${toolFn.function.name}`)

                if(toolFn.function.name === 'getDateTime'){
                    const exec = getDate()
                    ctx.push(firstCall.choices[0].message)
                    ctx.push({
                        role: 'tool',
                        content: exec.toString(),
                        tool_call_id: toolFn.id
                    })
                }
                if(toolFn.function.name === 'getBookingStatus'){
                    const args = JSON.parse(toolFn.function.arguments)
                    const exec = await getBookingStatus(bokingColl, args.bookingId)
                    ctx.push(firstCall.choices[0].message)
                    ctx.push({
                        role: 'tool',
                        content: exec,
                        tool_call_id: toolFn.id
                    })
                }
                if(toolFn.function.name === 'sendEmailToReception'){
                    const args = JSON.parse(toolFn.function.arguments)
                    const exec = await sendEmailToReception(args.content)
                    ctx.push(firstCall.choices[0].message)
                    ctx.push({
                        role: 'tool',
                        content: exec,
                        tool_call_id: toolFn.id
                    })
                }

                const nextCall = await openaiClient.chat.completions.create({
                    model: __GPT_MODEL__,
                    messages: ctx
                })
                console.log(`
                    ${JSON.stringify(nextCall.choices[0].message)}
                    -------------------------------------------
                    ${relevantMetadata}
                `)
            }else {
                console.log(`
                    ${JSON.stringify(firstCall.choices[0].message)}
                    -------------------------------------------
                    ${relevantMetadata}
                `)
            }
        }
    }
}


const getBookingStatus = async(coll: Collection, bookingId: string) => {
    console.log(bookingId)
    const res = await coll.findOne({"data.id":bookingId})
    return JSON.stringify({...res, error: false, errors: []})|| "Reservation not found."
}
const getDate = () => new Date()
const sendEmailToReception = (content: string) => {
    console.log(`sending email to reception - content ${content}`)
    return 'Email byl odeslán'
}