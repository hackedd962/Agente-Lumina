import { StreamingTextResponse, LangChainStream, Message } from 'ai';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { ConversationalRetrievalQAChain } from 'langchain/chains';
import { vectorStore } from '@/utils/openai';
import { NextResponse } from 'next/server';
import { BufferMemory } from "langchain/memory";
import { PromptTemplate } from "langchain/prompts";

export async function POST(req: Request) {
    try {
        const { stream, handlers } = LangChainStream();
        const body = await req.json();
        const messages: Message[] = body.messages ?? [];
        const question = messages[messages.length - 1].content;

        // 1. Forzamos el modelo gpt-4o-mini y bajamos la temperatura para evitar invenciones
        const model = new ChatOpenAI({
            modelName: "gpt-4o-mini", 
            temperature: 0.1, 
            streaming: true,
            callbacks: [handlers],
        });

        const retriever = vectorStore().asRetriever({ 
            "searchType": "mmr", 
            "searchKwargs": { "fetchK": 10, "lambda": 0.25 } 
        });

        // 2. Definimos el prompt estricto para el RAG
        // Nota: Usamos {context} y {question} que son las variables estándar de LangChain
        const qaPrompt = PromptTemplate.fromTemplate(`
Eres un asistente virtual de CFE para atención a clientes.
Tu ÚNICA fuente de información es el CONTEXTO proporcionado abajo.
Reglas ESTRICTAS:
1. Responde ÚNICAMENTE usando la información del CONTEXTO.
2. Si el CONTEXTO no contiene la respuesta, di exactamente: "No tengo esa información en mis documentos. Por favor, contacta a un asesor de CFE para ayudarte."
3. NUNCA inventes datos, tarifas, plazos o procedimientos que no estén en el CONTEXTO.
4. Responde en español, de forma clara y breve.

CONTEXTO:
{context}

PREGUNTA: {question}
RESPUESTA:`);

        // 3. Creamos la cadena inyectando el prompt personalizado
        // En la versión actual de LangChain, se usa qaChainOptions para el prompt final
        const conversationChain = ConversationalRetrievalQAChain.fromLLM(model, retriever, {
            memory: new BufferMemory({
              memoryKey: "chat_history",
            }),
            qaChainOptions: {
                type: "stuff",
                prompt: qaPrompt,
            },
        });

        // 4. Invocamos la cadena
        await conversationChain.invoke({
            "question": question
        });

        return new StreamingTextResponse(stream);
    }
    catch (e) {
        console.error("Error en la cadena RAG:", e);
        return NextResponse.json({ message: 'Error Processing' }, { status: 500 });
    }
}