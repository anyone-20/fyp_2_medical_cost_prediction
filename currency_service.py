# currency_service.py

from __future__ import annotations

from typing import Any

import requests


API_URL = (
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
    """

    if amount < 0:
        raise ValueError(
            "The amount cannot be negative."
        )

    if not api_key:
        raise ValueError(
            "EXCHANGE_RATE_API_KEY is missing."
        )

    from_currency = (
        from_currency
        .strip()
        .upper()
    )

    to_currency = (
        to_currency
        .strip()
        .upper()
    )

    if from_currency == to_currency:
        return {
            "original_amount": float(amount),
            "converted_amount": float(amount),
            "from_currency": from_currency,
            "to_currency": to_currency,
            "rate": 1.0,
            "last_updated": None,
            "next_update": None,
        }

    url = API_URL.format(
        api_key=api_key,
        base_currency=from_currency,
    )

    try:
        response = requests.get(
            url,
            timeout=10,
        )

        response.raise_for_status()

    except requests.Timeout as error:
        raise RuntimeError(
            "The currency service timed out."
        ) from error

    except requests.RequestException as error:
        raise RuntimeError(
            "Unable to connect to the currency service."
        ) from error

    try:
        data = response.json()

    except ValueError as error:
        raise RuntimeError(
            "The currency service returned invalid data."
        ) from error

    if data.get("result") != "success":
        error_type = data.get(
            "error-type",
            "unknown-error",
        )

        raise RuntimeError(
            f"ExchangeRate-API error: {error_type}"
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
        "original_amount": float(amount),
        "converted_amount": converted_amount,
        "from_currency": from_currency,
        "to_currency": to_currency,
        "rate": rate,
        "last_updated": data.get(
            "time_last_update_utc"
        ),
        "next_update": data.get(
            "time_next_update_utc"
        ),
    }
