import sys
import os
import re
from dotenv import load_dotenv

# Add project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from agents.faq_retriever import retrieve_faq_answer
from agents.forecasting_agent import ForecastingAgent
from agents.market_watcher_agent import MarketWatcherAgent
from agents.web_search_agent import query_web
from agents.summarizer_agent import SummarizerAgent  # For future summary usage

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
forecasting_agent = ForecastingAgent()  # Instantiate once and reuse

# Refined PromptTemplate for Query Classification
router_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a query classification agent. Classify the following query into one or more of the categories:
1. Forecasting → If the query asks for predictions (e.g., future demand, trends, projections)
2. MarketWatcher → If the query asks for current beauty product trends or competitor updates
3. WebSearch → If the query is about new trends, campaigns, or online brand visibility
4. FAQs → If the query is a general question about Nykaa, product usage, or returns

Query: "{query}"

Return only the category names: Forecasting, MarketWatcher, WebSearch, FAQs.
"""
)

# Create the LLMChain for classification
query_router = LLMChain(llm=llm, prompt=router_prompt)

# === Core Classification ===
def classify_query(query):
    return query_router.run(query).strip()

# === Decompose compound queries ===
def decompose_query(query):
    delimiters = [" and ", ",", "\n", " also ", " plus "]
    pattern = "|".join(map(re.escape, delimiters))
    return [q.strip() for q in re.split(pattern, query) if q.strip()]

# === Intelligent Router with Specific Product Handling ===
def route_and_answer(query):
    classification = classify_query(query)
    print(f"\n🔍 Classification: {classification}")

    answers = []
    forecasting_response = ""
    marketwatcher_response = ""
    websearch_response = ""
    faq_response = ""

    # === Forecasting Agent ===
    if "Forecasting" in classification:
        forecasting_response = forecasting_agent.forecast(query)
        answers.append(f"📈 Forecasting Agent:\n{forecasting_response}")

    if "FAQs" in classification:
        faq_response = retrieve_faq_answer(query)
        answers.append("💡 FAQs Agent:\n" + faq_response)

    if "MarketWatcher" in classification:
        market_agent = MarketWatcherAgent()
        marketwatcher_response = market_agent.compare_product(product_name=query, region="West")
        answers.append("📊 MarketWatcher Agent:\n" + marketwatcher_response)

    if "WebSearch" in classification:
        websearch_response = query_web(query)
        answers.append("🌐 WebSearch Agent:\n" + websearch_response)

    if not answers:
        return "❓ Sorry, I couldn’t route this query. Try rephrasing."

    return "\n---\n".join(answers)

# === Master Orchestrator ===
def detect_and_route(query):
    sub_queries = decompose_query(query)
    responses = []

    print("\n🧹 Sub-queries Detected:")
    for sub in sub_queries:
        print(f"→ {sub}")
        res = route_and_answer(sub)
        responses.append(f"\n🧠 Query: {sub}\n{res}")

    return "\n\n==============================\n".join(responses)

# === ✅ TEST ===
if __name__ == "__main__":
    test_queries = [
        #"Forecast Nykaa Shampoo demand in Karnataka next month",
        #"How much Nykaa SKINRX Vitamin C Serum will be needed in South in 2025-08-10?",
        #"Tell me the forecasted units for Nykaa Matte Lipstick in Karnataka for next month.",
        #"What is the market strategy for Nykaa Black Magic Liquid Eyeliner in West India?",
        #"What are the competitors of Nykaa primer in India?",
        #"How to return a product on Nykaa?"
        "Forecast Nykaa Shampoo demand in Karnataka next month and tell me who its main competitors are"
        "Forecast Nykaa Cosmetics Get Cheeky Blush Stick demand in Karnataka next month and tell me who its main competitors are"
        
    
    ]

    for query in test_queries:
        print("\n==============================")
        print(f"🔍 Original Query: {query}")
        result = detect_and_route(query)
        print(f"\n✅ Final Output:\n{result}")
        print("==============================\n")
