# myskills.py
import requests
import os

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
        

# graph_calendar_plugin.py
class GraphCalendarPlugin:
    """
    A simple plugin to demonstrate how to call Microsoft Graph's
    /me/events endpoint using an existing access token.
    """

    def __init__(self, access_token: str):
        self.access_token = access_token

    @kernel_function
    async def list_calendar_events(
        self,
        top: Annotated[int, "Number of events to fetch"] = 5
    ) -> str:
        """
        Lists the upcoming events in the user's primary calendar.
        Example usage:
          GraphCalendarPlugin.list_calendar_events "3"
        to get the next 3 events.
        """
        endpoint = f"https://graph.microsoft.com/v1.0/me/events?$top={top}&$orderby=start/dateTime"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        resp = requests.get(endpoint, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            events = data.get("value", [])
            if not events:
                return "No upcoming events found."

            lines = []
            for e in events:
                subject = e["subject"]
                start = e["start"].get("dateTime")
                end = e["end"].get("dateTime")
                lines.append(f"Event: {subject}\n Start: {start}\n End:   {end}\n")
            return "\n".join(lines)
        else:
            return f"Failed to fetch calendar events: {resp.status_code} - {resp.text}"

    @kernel_function
    async def create_calendar_event(
        self,
        subject: Annotated[str, "Event subject/title"],
        start_datetime: Annotated[str, "Event start date/time in ISO8601 format"],
        end_datetime: Annotated[str, "Event end date/time in ISO8601 format"]
    ) -> str:
        """
        Creates a new event in the user's primary calendar.
        Example call:
          GraphCalendarPlugin.create_calendar_event
            "Team Meeting" "2025-05-01T10:00:00" "2025-05-01T11:00:00"
        """
        endpoint = "https://graph.microsoft.com/v1.0/me/events"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        event_data = {
            "subject": subject,
            "start": {
                "dateTime": start_datetime,
                "timeZone": "UTC"  # or your local time zone
            },
            "end": {
                "dateTime": end_datetime,
                "timeZone": "UTC"
            }
        }

        resp = requests.post(endpoint, headers=headers, json=event_data)
        if resp.status_code in [200, 201]:
            created = resp.json()
            return f"Event created! ID: {created['id']}"
        else:
            return f"Failed to create event: {resp.status_code} - {resp.text}"

