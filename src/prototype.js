import "dotenv/config";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { YoutubeLoader } from "langchain/document_loaders/web/youtube";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { AzureChatOpenAI } from "@langchain/openai";
import { AzureOpenAIEmbeddings } from "@langchain/azure-openai";  
import {AzureAISearchVectoreStore, AzureAISearchQueryType} from "@langchain/community/vectorstores/azure_aisearch";

const YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=gfQhxffAuII";
const QUESTION = "Who do they think will win the main event?";

// Load documents ------------------------------------------------------------

console.log("Loading documents...");

const loader = YoutubeLoader.createFromUrl(YOUTUBE_VIDEO_URL, {
  language: "en",
  addVideoInfo: true,
});
const rawDocuments = await loader.load();
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 2000,
  chunkOverlap: 400,
});
const documents = await splitter.splitDocuments(rawDocuments);

console.log("Split Documents: ", documents)


// Init models and DB --------------------------------------------------------

console.log("Initializing models and DB...");

const embeddings = new AzureOpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY,
});


console.log("Embedding documents...");

const vectorStore = await FaissStore.fromDocuments({documents, embeddings });


const model = new AzureChatOpenAI();


// Run the chain -------------------------------------------------------------

console.log("Running the chain...");

const questionAnsweringPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "Answer the user's question using only the sources below:\n\n{context}",
  ],
  ["human", "{input}"],
]);
const combineDocsChain = await createStuffDocumentsChain({
  prompt: questionAnsweringPrompt,
  llm: model,
});
const chain = await createRetrievalChain({
  retriever: vectorStore.asRetriever(),
  combineDocsChain,
});
const stream = await chain.stream({ input: QUESTION });

// Print the result ----------------------------------------------------------

console.log(`Answer for the question "${QUESTION}":\n`);
for await (const chunk of stream) {
  process.stdout.write(chunk.answer ?? "");
}
console.log();
