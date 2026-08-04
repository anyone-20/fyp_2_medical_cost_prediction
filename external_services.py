from __future__ import annotations

from typing import Any

import requests
from google import genai


EXCHANGE_RATE_URL = (
    "https://v6.exchangerate-api.com/v6/"
    "{api_key}/latest/{base_currency}"
)


def convert_currency(
    *,
    amount: float,
    from_currency: str,
    to_currency: str,
    api_key: str,
) -> dict[str, Any]:
    """
    Convert an amount using ExchangeRate-API.

    Returns the rate, converted amount and provider update time.
    """

    if amount < 0:
        raise ValueError(
            "Currency amount cannot be negative."
        )

    from_currency = from_currency.upper().strip()
    to_currency = to_currency.upper().strip()

    if from_currency == to_currency:
        return {
            "from_currency": from_currency,
            "to_currency": to_currency,
            "rate": 1.0,
            "original_amount": float(amount),
            "converted_amount": float(amount),
            "last_updated": None,
        }

    if not api_key:
        raise ValueError(
            "The currency API key is missing."
        )

    url = EXCHANGE_RATE_URL.format(
        api_key=api_key,
        base_currency=from_currency,
    )

    try:
        response = requests.get(
            url,
            timeout=10,
        )

        response.raise_for_status()

    except requests.RequestException as error:
        raise RuntimeError(
            "Unable to connect to the currency service."
        ) from error

    data = response.json()

    if data.get("result") != "success":
        error_type = data.get(
            "error-type",
            "unknown-error",
        )

        raise RuntimeError(
            f"Currency API error: {error_type}"
        )

    conversion_rates = data.get(
        "conversion_rates",
        {}
    )

    if to_currency not in conversion_rates:
        raise ValueError(
            f"Currency '{to_currency}' is not supported."
        )

    rate = float(
        conversion_rates[to_currency]
    )

    converted_amount = (
        float(amount) * rate
    )

    return {
        "from_currency": from_currency,
        "to_currency": to_currency,
        "rate": rate,
        "original_amount": float(amount),
        "converted_amount": converted_amount,
        "last_updated": data.get(
            "time_last_update_utc"
        ),
    }


def create_gemini_client(
    api_key: str,
):
    """
    Create a Gemini client.
    """

    if not api_key:
        raise ValueError(
            "The Gemini API key is missing."
        )

    return genai.Client(
        api_key=api_key
    )


def generate_chatbot_response(
    *,
    client,
    user_message: str,
    chat_history: list[dict[str, str]],
    prediction_context: dict[str, Any] | None = None,
) -> str:
    """
    Generate a healthcare-cost explanation using Gemini.

    The chatbot is restricted to educational explanations
    and must not provide diagnosis or medical treatment advice.
    """

    if not user_message.strip():
        raise ValueError(
            "The chatbot message cannot be empty."
        )

    context_lines = []

    if prediction_context:
        context_lines = [
            (
                "Predicted inpatient cost in CNY: "
                f"{prediction_context.get('predicted_cny', 0):,.2f}"
            ),
            (
                "Selected display currency: "
                f"{prediction_context.get('display_currency', 'CNY')}"
            ),
            (
                "Converted prediction: "
                f"{prediction_context.get('converted_cost', 0):,.2f}"
            ),
            (
                "Model type: "
                "blended LightGBM and XGBoost regression model"
            ),
        ]

        top_factors = prediction_context.get(
            "top_factors",
            []
        )

        if top_factors:
            context_lines.append(
                "Top model factors: "
                + ", ".join(top_factors)
            )

    recent_history = chat_history[-8:]

    history_text = "\n".join(
        (
            f"{message['role']}: "
            f"{message['content']}"
        )
        for message in recent_history
    )

    system_context = """
You are an educational assistant embedded in a healthcare-cost
prediction application.

Your duties:
1. Explain model predictions in clear, simple language.
2. Explain what model features and SHAP contributions mean.
3. Explain currency conversion and clarify that exchange rates change.
4. Never diagnose diseases.
5. Never recommend medication, treatment or emergency action based only
   on this machine-learning prediction.
6. Never claim the predicted cost is guaranteed.
7. State that the output is an estimate based on historical survey data.
8. Do not invent model metrics, features or patient information.
9. Encourage users to consult qualified medical or financial
   professionals for personal decisions.
10. Keep responses concise and directly related to this application.
"""

    full_prompt = f"""
{system_context}

Current application context:
{chr(10).join(context_lines)}

Recent conversation:
{history_text}

User message:
{user_message}
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=full_prompt,
        store=False,
    )

    response_text = getattr(
        interaction,
        "output_text",
        None,
    )

    if not response_text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return response_text.strip()
