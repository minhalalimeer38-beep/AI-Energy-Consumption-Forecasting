import os
from config import PROJECT_DIR

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from ai_assistant import (
    get_consumption_summary,
    get_anomaly_summary,
    get_peak_usage_summary
)

from rag import retrieve_context


MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

def create_energy_agent(
    consumption_df=None,
    anomaly_df=None
):
    """
    Create the Agentic AI assistant.

    The agent can:
    - Retrieve energy knowledge using RAG
    - Analyze consumption data
    - Analyze anomalies
    - Identify peak usage
    """

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME
    )

    @tool
    def search_energy_knowledge(
        question: str
    ) -> str:
        """
        Search the project's energy knowledge base
        and return relevant information.
        """

        context = retrieve_context(
            question
        )

        if not context:

            return (
                "No relevant information was found "
                "in the energy knowledge base."
            )

        return context

    @tool
    def analyze_consumption() -> str:
        """
        Analyze the user's current electricity
        consumption data.
        """

        if consumption_df is None:

            return (
                "No user electricity data is currently "
                "available."
            )

        return get_consumption_summary(
            consumption_df,
            "Zone 1 Power Consumption"
        )

    @tool
    def analyze_anomalies() -> str:
        """
        Analyze detected electricity anomalies.
        """

        if anomaly_df is None:

            return (
                "No anomaly data is currently available."
            )

        return get_anomaly_summary(
            anomaly_df
        )

    @tool
    def analyze_peak_usage() -> str:
        """
        Analyze the user's electricity peak usage.
        """

        if consumption_df is None:

            return (
                "No user electricity data is currently "
                "available."
            )

        return get_peak_usage_summary(
            consumption_df,
            "Zone 1 Power Consumption"
        )

    system_prompt = """
You are the AI Energy Assistant for an
electricity consumption forecasting and
anomaly detection system.

Your responsibilities:

1. Answer questions about electricity consumption.
2. Use the consumption analysis tool when the
   user asks about their actual data.
3. Use the anomaly analysis tool when the
   user asks about unusual consumption.
4. Use the peak usage tool when the user asks
   about peak hours or high-usage periods.
5. Use the energy knowledge search tool when the
   question requires general energy-efficiency,
   electricity, building, or energy-management
   knowledge.
6. Clearly distinguish between:
   - information calculated from the user's data
   - general information retrieved from documents
7. Do not invent measurements or anomaly causes.
8. If the available data cannot answer a question,
   say so clearly.
9. Keep answers practical and easy to understand.

Language Rules:

10. Respond in the same language used by the user.
11. Support English, Urdu, and Roman Urdu.
12. If the user writes in Roman Urdu, respond in
    clear and natural Roman Urdu.
13. If the user writes in Urdu, respond in Urdu.
14. If the user writes in English, respond in English.
15. Do not switch languages unnecessarily.
16. Keep technical terms, dates, numbers, units,
    model names, and measurements accurate.
"""

    tools = [
        search_energy_knowledge,
        analyze_consumption,
        analyze_anomalies,
        analyze_peak_usage
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

    return agent

def ask_energy_agent(
    agent,
    question: str
) -> str:
    """
    Send a user question to the energy agent
    and return the final response.
    """

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    messages = result.get(
        "messages",
        []
    )

    if not messages:

        return (
            "I could not generate a response."
        )

    final_message = messages[-1]

    content = getattr(
        final_message,
        "content",
        ""
    )

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(
                        item.get("text", "")
                    )

            elif isinstance(item, str):
                text_parts.append(item)

        return "\n".join(
            part for part in text_parts if part
        )

    return str(content)