import ast
import operator
import requests

from langchain_core.tools import tool


# ============================================================
# WEATHER TOOL
# ============================================================

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a city.

    Use this tool when the user asks about current weather,
    temperature, wind speed, or weather conditions.

    Args:
        city: Name of the city, for example Lahore, Islamabad,
              Karachi, or London.
    """

    try:
        # Geocoding API
        geocode_url = "https://geocoding-api.open-meteo.com/v1/search"

        geocode_response = requests.get(
            geocode_url,
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json",
            },
            timeout=10,
        )

        geocode_response.raise_for_status()

        geocode_data = geocode_response.json()

        results = geocode_data.get("results")

        if not results:
            return f"ERROR: I could not find the city '{city}'."

        location = results[0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        city_name = location.get("name", city)
        country = location.get("country", "Unknown")

        # Weather API
        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_response = requests.get(
            weather_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "apparent_temperature,"
                    "weather_code,"
                    "wind_speed_10m"
                ),
                "timezone": "auto",
            },
            timeout=10,
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

        current = weather_data.get("current", {})

        temperature = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        apparent_temperature = current.get("apparent_temperature")
        wind_speed = current.get("wind_speed_10m")
        weather_code = current.get("weather_code")

        weather_description = weather_code_to_text(weather_code)

        return (
            f"Weather for {city_name}, {country}: "
            f"Temperature: {temperature}°C, "
            f"Feels like: {apparent_temperature}°C, "
            f"Humidity: {humidity}%, "
            f"Wind speed: {wind_speed} km/h, "
            f"Condition: {weather_description}."
        )

    except requests.exceptions.RequestException as exc:
        return f"ERROR: Weather service is unavailable. Details: {exc}"

    except Exception as exc:
        return f"ERROR: Unable to get weather information. Details: {exc}"


def weather_code_to_text(code):
    """
    Convert Open-Meteo weather codes into readable text.
    """

    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }

    return weather_codes.get(code, "Unknown weather condition")


# ============================================================
# CALCULATOR TOOL
# ============================================================

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_calculate(node):
    """
    Safely evaluate a mathematical AST.
    """

    if isinstance(node, ast.Expression):
        return safe_calculate(node.body)

    if isinstance(node, ast.Constant):

        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):

        operator_type = type(node.op)

        if operator_type not in _ALLOWED_OPERATORS:
            raise ValueError("Unsupported mathematical operator.")

        left = safe_calculate(node.left)
        right = safe_calculate(node.right)

        return _ALLOWED_OPERATORS[operator_type](left, right)

    if isinstance(node, ast.UnaryOp):

        operator_type = type(node.op)

        if operator_type not in _ALLOWED_OPERATORS:
            raise ValueError("Unsupported unary operator.")

        operand = safe_calculate(node.operand)

        return _ALLOWED_OPERATORS[operator_type](operand)

    raise ValueError("Invalid mathematical expression.")


@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations.

    Use this tool when the user asks for arithmetic,
    percentages, multiplication, division, addition,
    subtraction, powers, or similar calculations.

    Example:
        25 * 40
        1200 * 0.25
        (50 + 20) / 2

    Args:
        expression: Mathematical expression.
    """

    try:
        expression = expression.strip()

        if not expression:
            return "ERROR: Empty mathematical expression."

        tree = ast.parse(expression, mode="eval")

        result = safe_calculate(tree)

        return f"Calculation result: {result}"

    except ZeroDivisionError:
        return "ERROR: Cannot divide by zero."

    except Exception as exc:
        return f"ERROR: Invalid calculation. Details: {exc}"


# ============================================================
# TOOL LIST
# ============================================================

TOOLS = [
    get_weather,
    calculator,
]