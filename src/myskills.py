# myskills.py
import requests
import os, httpx, asyncio

from geopy.geocoders import Nominatim
from dotenv import load_dotenv
from typing import Annotated
from semantic_kernel.functions import kernel_function

# Load environment variables from .env file
load_dotenv()

nominatim_name = os.getenv("NOMINATIM_USERNAME", "")

class WeatherPlugin:
    def __init__(self):
        # For Open-Meteo + Nominatim, we don’t need an API key
        pass

    @kernel_function
    async def get_weather_by_city(self, city: Annotated[str, "City to fetch weather for"]) -> str:
        """
        Gets current weather from Open-Meteo by first geocoding the city name
        with Nominatim (OpenStreetMap).
        Example call in a prompt or code: 'WeatherPlugin.get_weather_by_city "Chicago"'
        """

        if not city:
            return "You must provide a city name."
        print("Fetching weather for city:", city)

        # Step 1: Geocode city → lat/long via Nominatim
        geolocator = Nominatim(user_agent=nominatim_name)
        location = geolocator.geocode(city)

        lat = location.latitude
        lon = location.longitude

        # Step 2: Fetch weather from Open-Meteo
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": "true"
        }
        weather_resp = requests.get(weather_url, params=weather_params)

        if weather_resp.status_code == 200:
            data = weather_resp.json()
            current = data.get("current_weather", {})
            temperature = current.get("temperature")
            windspeed = current.get("windspeed")
            return (f"Current weather in {city}: {temperature}°C, windspeed {windspeed} km/h.")
        else:
            return f"Open-Meteo API request failed with status {weather_resp.status_code}"
        

class MathPlugin:
    def __init__(self):
        pass

    @kernel_function
    async def add(self, 
                  a: Annotated[float, "First number"], 
                  b: Annotated[float, "Second number"]) -> str:
        """
        Returns the sum of two numbers.
        Example: 'MathPlugin.add 3 5' → "Result: 8.0"
        """
        result = a + b
        return f"Result: {result}"

    @kernel_function
    async def subtract(self, 
                       a: Annotated[float, "First number"], 
                       b: Annotated[float, "Second number"]) -> str:
        """
        Returns the difference of two numbers.
        Example: 'MathPlugin.subtract 10 4' → "Result: 6.0"
        """
        result = a - b
        return f"Result: {result}"

    @kernel_function
    async def multiply(self, 
                       a: Annotated[float, "First number"], 
                       b: Annotated[float, "Second number"]) -> str:
        """
        Returns the product of two numbers.
        Example: 'MathPlugin.multiply 3 7' → "Result: 21.0"
        """
        result = a * b
        return f"Result: {result}"

    @kernel_function
    async def divide(self, 
                     a: Annotated[float, "Dividend"], 
                     b: Annotated[float, "Divisor"]) -> str:
        """
        Returns the quotient of two numbers.
        Example: 'MathPlugin.divide 8 2' → "Result: 4.0"
        """
        if b == 0:
            return "Error: Division by zero is not allowed."
        result = a / b
        return f"Result: {result}"


class InternetSearchPlugin:
    def __init__(self, brave_api_key: str | None = None):
        # Accept key explicitly or fall back to env-var
        self._key = brave_api_key or os.getenv("BRAVE_API_KEY")
        if not self._key:
            raise ValueError(
                "No API key provided. "
                "Pass it to EdgeSearchPlugin(...) or set ***_API_KEY in your environment."
            )
        self._endpoint = "https://api.search.brave.com/res/v1/web/search"
        self._headers  = {"X-Subscription-Token": self._key}
        self._client   = httpx.AsyncClient(timeout=15)

    # ---- internal helper --------------------------------------------------
    async def _search_brave(self, query: str, count: int = 5) -> list[dict]:
        params = {"q": query, "count": count, "safesearch": "strict"}
        r = await self._client.get(self._endpoint, headers=self._headers, params=params)
        r.raise_for_status()
        items = r.json().get("web", {}).get("results", [])
        return [
            {"title": i["title"], "url": i["url"], "snippet": i["description"]}
            for i in items
        ]

    @kernel_function
    async def search(
        self,
        query: Annotated[str, "Text to search for"],
        count: Annotated[int, "Number of results to return"] = 5,
    ) -> str:
        if not query:
            return "You must provide a search query."
        try:
            results = await self._search_brave(query, count)
        except Exception as exc:
            return f"Search failed: {exc}"
        if not results:
            return "No results found."
        return "\n".join(
            f"{idx}. {r['title']} – {r['url']}\n   {r['snippet']}"
            for idx, r in enumerate(results, 1)
        )