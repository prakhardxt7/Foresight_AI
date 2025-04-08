import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSequence
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Initialize Embedding Model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector store path
CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../embeddings/chroma"))

# Load or create vector store from FAQs
def load_or_create_vectorstore(csv_path):
    index_path = os.path.join(CHROMA_PATH, "index")
    if os.path.exists(index_path):
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

    df = pd.read_csv(csv_path)
    docs = []
    for i, row in df.iterrows():
        if pd.notnull(row['question']) and pd.notnull(row['answer']):
            content = f"Q: {row['question'].strip()}\nA: {row['answer'].strip()}"
            docs.append(Document(page_content=content))

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(docs)

    vectordb = Chroma.from_documents(docs, embedding=embedding, persist_directory=CHROMA_PATH)
    return vectordb

# Load vectorstore
faq_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/faqs.csv"))
vectorstore = load_or_create_vectorstore(faq_csv_path)

# Retrieval function with enhanced Gemini business assistant prompt
def retrieve_faq_answer(query, similarity_threshold=0.8):
    results = vectorstore.similarity_search_with_relevance_scores(query, k=1)

    if results and results[0][1] >= similarity_threshold:
        return results[0][0].page_content

    # If no match, use Gemini to generate a response and store it
    prompt = PromptTemplate.from_template("""
You are a highly intelligent business assistant for Nykaa's customer support team. Your task is to help answer customer queries with clear, concise, and actionable responses.

1. **General FAQ**: Answer the question based on Nykaa FAQs. If the question is directly related to return, refund, delivery, or product usage, provide a straightforward response.
2. **Demand Forecasting**: If the query contains terms like "forecast", "sales", "demand", or "project", generate a detailed forecast based on product trends and regional sales. Include the forecasted units, key factors influencing demand, and a recommendation for inventory planning.
3. **Competitor Analysis**: If the query involves a competitor (e.g., "Purplle", "Sugar Cosmetics", "Mamaearth"), provide a comparison between Nykaa's products and the competitor’s products. Mention pricing, sales trends, and any insights that can help Nykaa strategize.
4. **Strategic Insights**: If the query involves broader business strategy (e.g., "pricing", "promotion", "marketing"), provide actionable insights that can help Nykaa improve its competitive position.

**Important**: For queries not directly related to FAQs, ensure that the answer includes **business insights**, **recommendations**, or **strategic suggestions**.

Input Query: {query}

Provide a well-structured response by following the appropriate section(s) above. If no FAQ matches, provide insights based on the query type.

Return the complete response without additional explanations.
""")
    chain = prompt | llm
    answer = chain.invoke({"query": query}).content.strip()

    # Store new Q&A
    new_entry = pd.DataFrame([[query, answer]], columns=["question", "answer"])
    new_entry.to_csv(faq_csv_path, mode="a", index=False, header=False)

    # Add to vectorstore
    content = f"Q: {query}\nA: {answer}"
    vectorstore.add_documents([Document(page_content=content)])

    return f"Q: {query}\nA: {answer}"

# Test script
if __name__ == "__main__":
    test_queries = [
        "How can I return a damaged lipstick?",
        "Can I cancel my Nykaa order before it is shipped?",
        "What's the right way to use hyaluronic acid serum?",
        "What is the forecast for Nykaa SKINRX Vitamin C Serum in North India for next month?",
        "How does Nykaa's pricing compare to Purplle's for skincare products?"
    ]
    for q in test_queries:
        print(f"\n🔍 Query: {q}")
        print(retrieve_faq_answer(q))
