def get_active_plugins(plugins: dict) -> dict:
    """
    Check which plugins are enabled and return a dictionary of plugin functions
    that should be attached to the kernel.
    For now, these are placeholders.
    """
    active = {}
    if plugins.get("location"):
        active["location_plugin"] = lambda: "Location plugin activated"  # Replace with actual function
    if plugins.get("weather"):
        active["weather_plugin"] = lambda: "Weather plugin activated"  # Replace with actual function
    if plugins.get("time"):
        active["time_plugin"] = lambda: "Time plugin activated"  # Replace with actual function
    return active
