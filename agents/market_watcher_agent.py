import os
import pandas as pd
import difflib
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Gemini Setup ===
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# === Refined Prompt for Business Summary ===
summary_prompt = PromptTemplate.from_template("""
You are a senior market strategist for a retail beauty brand.
Analyze the comparative data between Nykaa's product and its competitor in terms of sales performance, pricing, marketing strategies, and social media impact.

Nykaa Product: {nykaa_product}
Competitor Product: {competitor_product}
Region: {region}


Nykaa → Price: ₹{nykaa_price}, Sales: {nykaa_sales} units, Marketing: ₹{nykaa_marketing}, Social Influence: {nykaa_social}
Competitor → Price: ₹{comp_price}, Sales: {comp_sales} units, Marketing: ₹{comp_marketing}, Social Influence: {comp_social}

Focus your analysis on the following:
1. **Sales Comparison**: Who is leading in sales and why.
2. **Pricing & Marketing Effectiveness**: What are the price and marketing strategies of both brands, and how effective are they?
3. **Social Media Influence**: Analyze the impact of social media on the product's reach and sales.
4. **Strategic Recommendations**: Provide actionable insights for Nykaa’s offline store or digital strategy to compete better in the market.

Format your output as follows:

📌 **Business Summary**:
- [Actionable insight based on the comparison]
- [Strategic recommendations for Nykaa]

📊 **Competitors Mentioned**:
- [List of competitors or brands]
If no valid insight, return "No concrete competitor/product insight found."
""")

class MarketWatcherAgent:
    def __init__(
        self,
        nykaa_path="data/Nykaa_Enriched_Dataset_old.csv",
        competitor_path="data/Competitor_Dataset_old.csv",
        matched_path="data/matched_products.csv"
    ):
        self.nykaa_df = pd.read_csv(nykaa_path)
        self.competitor_df = pd.read_csv(competitor_path)
        self.matched_df = pd.read_csv(matched_path)

        # Preprocess product names and normalize case
        self.nykaa_df['Product_Name'] = self.nykaa_df['Product_Name'].str.lower().str.strip()
        self.competitor_df['Product_Name'] = self.competitor_df['Product_Name'].str.lower().str.strip()
        self.matched_df['Nykaa_Product_Name'] = self.matched_df['Nykaa_Product_Name'].str.lower().str.strip()
        self.matched_df['Competitor_Product_Name'] = self.matched_df['Competitor_Product_Name'].str.lower().str.strip()

        self.known_nykaa_products = self.matched_df['Nykaa_Product_Name'].unique().tolist()

    def _fuzzy_match_product(self, query: str):
        query = query.lower().strip()
        matches = difflib.get_close_matches(query, self.known_nykaa_products, n=1, cutoff=0.4)
        return matches[0] if matches else None

    def compare_product(self, product_name: str, region: str, date: str = None) -> str:
        # Fuzzy match product name
        matched_product = self._fuzzy_match_product(product_name)
        if not matched_product:
            return f"❌ No matching competitor found for '{product_name}'."

        region = region.capitalize()

        # Match Nykaa product with competitor
        match_row = self.matched_df[self.matched_df['Nykaa_Product_Name'] == matched_product]
        if match_row.empty:
            return f"❌ No competitor mapping for matched product '{matched_product}'."

        competitor_product = match_row.iloc[0]['Competitor_Product_Name']

        # Filter data for competitor and Nykaa product in the specified region
        comp_data = self.competitor_df[
            (self.competitor_df['Product_Name'] == competitor_product) &
            (self.competitor_df['Region'] == region)
        ]
        nykaa_data = self.nykaa_df[
            (self.nykaa_df['Product_Name'] == matched_product) &
            (self.nykaa_df['Region'] == region)
        ]

        if date:
            comp_data = comp_data[comp_data['Date'] == date]
            nykaa_data = nykaa_data[nykaa_data['Date'] == date]
        else:
            comp_data = comp_data.sort_values('Date', ascending=False).head(1)
            nykaa_data = nykaa_data.sort_values('Date', ascending=False).head(1)

        if comp_data.empty or nykaa_data.empty:
            return f"❌ Insufficient data for comparison in {region}."

        # Extract data for competitor and Nykaa product
        c = comp_data.iloc[0]
        n = nykaa_data.iloc[0]

        # Format input for the summary
        input_text = summary_prompt.format(
            nykaa_product=n['Product_Name'].title(),
            competitor_product=c['Product_Name'].title(),
            region=region,
            date=c['Date'],
            nykaa_price=round(n['Price_At_Time'], 2),
            nykaa_sales=int(n['Sales_Units']),
            nykaa_marketing=round(n['Marketing_Spend'], 2),
            nykaa_social=round(n['Social_Media_Influence_Score'], 2),
            comp_price=round(c['Price_At_Time'], 2),
            comp_sales=int(c['Sales_Units']),
            comp_marketing=round(c['Marketing_Spend'], 2),
            comp_social=round(c['Social_Media_Influence_Score'], 2)
        )

        # Generate the summary using Gemini LLM
        try:
            summary = llm.invoke(input_text).content.strip()
        except:
            summary = "📌 Note: Unable to generate insights due to API issue."

        # Return formatted result
        return f"""
📊 **Market Comparison:**
- **Nykaa Product:** {n['Product_Name'].title()} 
- **Competitor Product:** {c['Product_Name'].title()}
- **Region:** {region}
- 

📝 **Gemini Summary:**
{summary}
"""

# === ✅ Test ===
if __name__ == "__main__":
    agent = MarketWatcherAgent()

    test_queries = [
        ("Nykaa Black Magic Liquid Eyeliner", "West"),
        ("Nykaa SKINRX Vitamin C Serum", "North"),
        ("Nykaa Naturals Tea Tree & Neem Face Wash", "South"),
        ("Identify the key competitors of Nykaa SKINRX Vitamin C Serum", "North")
    ]

    for i, (product, region) in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {product} in {region}")
        print(agent.compare_product(product, region))