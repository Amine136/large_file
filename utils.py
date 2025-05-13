from unsloth import FastLanguageModel
import json
import os
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from pydantic import BaseModel, Field

#Model 1

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/model1/checkpoint-201",  # Path to your checkpoint
    max_seq_length = 2048,  # Adjust based on your training setup
    dtype = None,  # Will automatically detect
    load_in_4bit = True,  # Keep as True since you used 4-bit quantization
    # token = "your_hf_token_here", # Only needed if using gated models
)


prompt = """
Below is an analysis that requires biological interpretation.

### Instruction:
Interpret this biology analysis.

### Input:
{}

### Response:
{}"""


def generate_response(input_text):
    # Enable faster inference
    FastLanguageModel.for_inference(model)

    # Format the input
    formatted_input = prompt.format(input_text, "")

    # Tokenize input
    inputs = tokenizer(
        [formatted_input],
        return_tensors="pt"
    ).to("cuda")

    # Generate output
    output_ids = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and clean output
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only the response part

    response = full_text.split("### Response:")[1].strip()
    # Remove any remaining special tags and whitespace
    return response.replace("</s>", "").strip()


def _generate_and_save(path, text):
    # Use your existing function to generate the answer
    answer = generate_response(text)

    # Create a dictionary with both the input text and generated answer
    data = {
        "text": text,
        "answer": answer
    }

    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the dictionary as a JSON file
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Saved response to {path}")







#Model 2


embedding_model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')


class BioSentVecEmbedding(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=False  # Chroma expects lists, not tensors
        ).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()


embedding = BioSentVecEmbedding(embedding_model)



llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)



# Load the existing Chroma vectorstore
vectorstore = Chroma(persist_directory="/model2/chroma_biomed", embedding_function=embedding)





# 1) two prompt‐texts (different for retriever vs. LLM)
retriever_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    You are a clinical information specialist.
    Reformulate the question into a short, keyword-rich search query to retrieve relevant information.
    Avoid Boolean operators like AND/OR. Use medically meaningful phrases.

    Original Question:
    {question}

    Search Query:

      """,
)

llm_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a medical AI assistant.
Use ONLY the information in the CONTEXT (from 'Principles of Internal Medicine') to answer the QUESTION.
Cite the context where appropriate.
If the answer isn’t in the context, respond:
"Based on the available context, there is no sufficient information to answer."

        CONTEXT:
        {context}

        QUESTION:
        {question}

        Answer:
        """,
)

# 2) Build an LLMChain that *rewrites* the user’s question into a search query
query_rewriter = LLMChain(llm=llm, prompt=retriever_prompt)




class TransformRetriever(BaseRetriever, BaseModel):
    base_retriever: BaseRetriever = Field(...)
    query_chain: LLMChain = Field(...)

    def get_relevant_documents(self, question: str):
        search_query = self.query_chain.run(question=question)
        print(f"Rewritten Query: {search_query}")
        return self.base_retriever.get_relevant_documents(search_query)

    async def aget_relevant_documents(self, question: str):
        # Implement async version if needed
        search_query = await self.query_chain.arun(question=question)
        return await self.base_retriever.aget_relevant_documents(search_query)



base_retriever = vectorstore.as_retriever(search_type="similarity", k=10)


# Then use it the same way:
smart_retriever = TransformRetriever(
    base_retriever=base_retriever,
    query_chain=query_rewriter
)


qa_chain = load_qa_chain(
    llm=llm,
    chain_type="stuff",
    prompt=llm_prompt
)

retrieval_qa = RetrievalQA(
    retriever=smart_retriever,
    combine_documents_chain=qa_chain
)
