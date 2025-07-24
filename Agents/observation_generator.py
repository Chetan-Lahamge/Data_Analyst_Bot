import pandas as pd
import config
import logging
from openai import OpenAI
from Agents.sql_query_generator_agent import get_few_shot_prompt, matches_few_shot_case

# Setup logging
logger = logging.getLogger(__name__)
client = OpenAI(api_key=config.OPENAI_API_KEY)

async def generate_observations(df: pd.DataFrame, user_query: str) -> str:
    """
    Generate business observations using LLM based on query and data.
    Uses few-shot prompt if applicable to guide the LLM reasoning.
    """
    try:
        if df is None or df.empty:
            return "The data for this query is not available for the selected time period."

        # Step 1: Get preview of actual data
        preview = df.head(12).to_string(index=False)
        col_summary = "\n".join([f"- {col}: {dtype}" for col, dtype in df.dtypes.items()])

        # Step 2: Check if the query matches a guided few-shot case
        if matches_few_shot_case(user_query):
            try:
                few_shot = get_few_shot_prompt(user_query).strip()
            except Exception as fs_err:
                logger.warning(f"Few-shot prompt not found: {fs_err}")
                few_shot = ""

            prompt = f"""
The user asked: "{user_query}"

Below is a snapshot of the actual data:

Table Preview (first 5 rows):
{preview}

Column Types:
{col_summary}

Use the reasoning format shown below to generate a detailed response:

{few_shot}

Now, provide a similar response using the above data.

⚠️ Important Instructions:
- ❌ Do not include SQL code or code blocks in the observations.
- ✅ Provide only business-friendly insights and explanations in plain language.
- Keep your tone clear, helpful, and concise.
"""
        else:
            # Generic observation fallback
            prompt = f"""
You are a data analyst. The user asked: "{user_query}"

Below is the actual data from the database:

Table Preview (first 5 rows):
{preview}

Column Types:
{col_summary}

Now write 4–5 clear business insights or observations based on the user's question and the actual data above.

Guidelines:
- Use natural, plain English.
- Mention patterns, trends, comparisons, or outliers.
- If data is inconclusive or incomplete, say so clearly.
- Recommend possible business actions if appropriate.
- Avoid SQL or technical jargon.
- ❌ Do not include SQL code or code blocks in the observations.
- ✅ Provide only business-friendly analysis and plain language insights.
- Respond only with bullet points.
"""

        # Step 4: Send prompt to GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error generating observations: {str(e)}")
        return "Observations could not be generated due to an error."
