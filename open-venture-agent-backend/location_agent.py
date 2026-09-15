import re
from typing import Literal, TypedDict

import httpx
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from langgraph.graph import END, START, StateGraph

SEARCH_RADIUS_METERS = 1500
OVERPASS_TIMEOUT_SECONDS = 45.0
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]
NOMINATIM_USER_AGENT = "open-venture-agent/1.0"
OVERPASS_HEADERS = {
    "User-Agent": "open-venture-agent/1.0 (open-venture-agent-backend)",
    "Accept": "application/json",
    "Referer": "https://open-venture-agent.local",
}
OVERPASS_UNAVAILABLE_EXPLANATION = (
    "Competitor density could not be verified due to a temporary map-data service "
    "outage. Location recommendation should be treated as provisional until this "
    "is re-checked."
)

CATEGORY_FILTERS: dict[str, str] = {
    "gym": '["leisure"="fitness_centre"]',
    "fitness": '["leisure"="fitness_centre"]',
    "coffee": '["amenity"="cafe"]',
    "cafe": '["amenity"="cafe"]',
    "restaurant": '["amenity"="restaurant"]',
    "clothing": '["shop"="clothes"]',
    "apparel": '["shop"="clothes"]',
    "shop": '["shop"]',
}


class LocationState(TypedDict, total=False):
    business_idea: str
    category: str
    location_name: str
    latitude: float
    longitude: float
    competitor_count: int | None
    commercial_count: int | None
    score: int | None
    status: str
    data_status: str
    result: dict


FILLER_WORDS = re.compile(
    r"\b(city|town|village|the middle of|middle of|area|region|district)\b",
    re.IGNORECASE,
)


def parse_business_idea(business_idea: str) -> tuple[str, str]:
    match = re.search(r"\bin\s+(.+)$", business_idea, flags=re.IGNORECASE)
    if match:
        location = match.group(1).strip()
        category = business_idea[: match.start()].strip()
        return category or business_idea, location
    return business_idea, business_idea


def detect_category_filter(category: str) -> str:
    category_lower = category.lower()
    for keyword, overpass_filter in CATEGORY_FILTERS.items():
        if keyword in category_lower:
            return overpass_filter
    return '["amenity"="restaurant"]'


def _simplify_location_query(query: str) -> str:
    simplified = FILLER_WORDS.sub("", query)
    return re.sub(r"\s+", " ", simplified).strip(" ,")


def _try_geocode(geolocator: Nominatim, query: str):
    if not query:
        return None
    try:
        return geolocator.geocode(query, language="en")
    except (GeocoderServiceError, GeocoderTimedOut, ValueError):
        return None


def _try_reverse_geocode(
    geolocator: Nominatim,
    latitude: float,
    longitude: float,
):
    """Resolve lat/lon to a human-readable address via Nominatim reverse."""
    try:
        return geolocator.reverse(
            (latitude, longitude),
            language="en",
            exactly_one=True,
            addressdetails=True,
        )
    except (GeocoderServiceError, GeocoderTimedOut, ValueError):
        return None


def _format_human_readable_address(
    reversed_location,
    fallback: str | None = None,
) -> str | None:
    """Build a short place label from reverse-geocode address parts."""
    if reversed_location is None:
        return fallback

    raw = reversed_location.raw or {}
    addr = raw.get("address") or {}

    locality_keys = (
        "neighbourhood",
        "suburb",
        "quarter",
        "city_district",
        "residential",
        "village",
        "town",
        "city",
        "municipality",
    )
    region_keys = ("county", "state_district", "state")

    parts: list[str] = []
    seen: set[str] = set()

    def _add(value: str | None) -> None:
        if not value:
            return
        key = value.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        parts.append(value.strip())

    for key in locality_keys:
        _add(addr.get(key))
        if len(parts) >= 2:
            break

    for key in region_keys:
        before = len(parts)
        _add(addr.get(key))
        if len(parts) > before:
            break

    _add(addr.get("country"))

    if parts:
        return ", ".join(parts)

    if reversed_location.address:
        return reversed_location.address
    return fallback


def _location_not_found_result(location_query: str) -> dict:
    return {
        "status": "location_not_found",
        "result": {
            "status": "location_not_found",
            "location_name": location_query,
            "coordinates": None,
            "competitor_count_nearby": None,
            "score": None,
            "data_status": "location_not_found",
            "data_confidence": "high",
            "note": None,
            "explanation": "Could not geocode the location from the business idea.",
        },
    }


def _coordinates_payload(
    latitude: float | None,
    longitude: float | None,
) -> dict | None:
    if latitude is None or longitude is None:
        return None
    return {
        "latitude": latitude,
        "longitude": longitude,
    }


def _overpass_unavailable_result(state: LocationState) -> dict:
    result = {
        "status": "overpass_unavailable",
        "location_name": state.get("location_name"),
        "coordinates": _coordinates_payload(
            state.get("latitude"),
            state.get("longitude"),
        ),
        "competitor_count_nearby": None,
        "score": None,
        "data_status": "overpass_unavailable",
        "data_confidence": "high",
        "note": None,
        "explanation": OVERPASS_UNAVAILABLE_EXPLANATION,
    }
    return {
        "competitor_count": None,
        "commercial_count": None,
        "score": None,
        "status": "overpass_unavailable",
        "data_status": "overpass_unavailable",
        "result": result,
    }


def geocode_location(state: LocationState) -> dict:
    category, location_query = parse_business_idea(state["business_idea"])
    geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)

    queries = [location_query]
    simplified = _simplify_location_query(location_query)
    if simplified and simplified != location_query:
        queries.append(simplified)

    location = None
    for query in queries:
        location = _try_geocode(geolocator, query)
        if location is not None:
            break

    if location is None:
        return _location_not_found_result(location_query)

    latitude = location.latitude
    longitude = location.longitude
    # Reverse geocode so coordinates map to a clear human-readable place name.
    reversed_location = _try_reverse_geocode(geolocator, latitude, longitude)
    location_name = _format_human_readable_address(
        reversed_location,
        fallback=location.address,
    )

    return {
        "category": category,
        "location_name": location_name,
        "latitude": latitude,
        "longitude": longitude,
        "status": "ok",
    }


def _overpass_count(query: str) -> int | None:
    """Return element count, or None if all Overpass mirrors fail (non-fatal)."""
    for url in OVERPASS_URLS:
        try:
            response = httpx.post(
                url,
                data={"data": query},
                headers=OVERPASS_HEADERS,
                timeout=OVERPASS_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            return len(data.get("elements", []))
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            continue
    return None


def find_nearby_places(state: LocationState) -> dict:
    lat = state["latitude"]
    lon = state["longitude"]
    category_filter = detect_category_filter(state["category"])

    competitor_query = f"""
    [out:json][timeout:25];
    (
      nwr{category_filter}(around:{SEARCH_RADIUS_METERS},{lat},{lon});
    );
    out center;
    """

    commercial_query = f"""
    [out:json][timeout:25];
    (
      nwr["shop"](around:{SEARCH_RADIUS_METERS},{lat},{lon});
      nwr["amenity"~"restaurant|cafe|bank|pharmacy|supermarket|fast_food|marketplace"](around:{SEARCH_RADIUS_METERS},{lat},{lon});
      nwr["leisure"="fitness_centre"](around:{SEARCH_RADIUS_METERS},{lat},{lon});
    );
    out center;
    """

    competitor_count = _overpass_count(competitor_query)
    commercial_count = _overpass_count(commercial_query)

    if competitor_count is None or commercial_count is None:
        return _overpass_unavailable_result(state)

    return {
        "competitor_count": competitor_count,
        "commercial_count": commercial_count,
        "data_status": "ok",
    }


def calculate_location_score(state: LocationState) -> dict:
    competitor_count = state["competitor_count"]
    commercial_count = state["commercial_count"]
    if competitor_count is None or commercial_count is None:
        return _overpass_unavailable_result(state)

    # Fewer direct competitors is better (up to 55 points).
    competitor_score = max(0, 55 - min(competitor_count, 11) * 5)

    # More nearby commercial activity is better (up to 45 points).
    commercial_score = min(45, commercial_count * 1.5)

    score = int(round(min(100, competitor_score + commercial_score)))

    explanation = (
        f"Within {SEARCH_RADIUS_METERS}m: {competitor_count} direct competitors and "
        f"{commercial_count} commercial POIs nearby. "
        f"Competition score: {competitor_score}/55, "
        f"commercial activity score: {int(commercial_score)}/45."
    )

    result = {
        "location_name": state["location_name"],
        "coordinates": _coordinates_payload(
            state["latitude"],
            state["longitude"],
        ),
        "competitor_count_nearby": competitor_count,
        "score": score,
        "data_status": "ok",
        "data_confidence": "high",
        "note": None,
        "explanation": explanation,
    }

    return {"score": score, "result": result}


def route_after_geocode(state: LocationState) -> Literal["find_nearby", "__end__"]:
    if state.get("status") == "location_not_found":
        return "__end__"
    return "find_nearby"


def route_after_nearby(
    state: LocationState,
) -> Literal["score", "__end__"]:
    if state.get("status") == "overpass_unavailable":
        return "__end__"
    return "score"


def build_location_graph():
    graph = StateGraph(LocationState)
    graph.add_node("geocode", geocode_location)
    graph.add_node("find_nearby", find_nearby_places)
    graph.add_node("score", calculate_location_score)
    graph.add_edge(START, "geocode")
    graph.add_conditional_edges(
        "geocode",
        route_after_geocode,
        {"find_nearby": "find_nearby", "__end__": END},
    )
    graph.add_conditional_edges(
        "find_nearby",
        route_after_nearby,
        {"score": "score", "__end__": END},
    )
    graph.add_edge("score", END)
    return graph.compile()


location_agent = build_location_graph()


def run_location_analysis(business_idea: str) -> dict:
    result = location_agent.invoke({"business_idea": business_idea})
    return result["result"]


if __name__ == "__main__":
    analysis = run_location_analysis("computer repair shop in alexandria city, washington dc")
    print(analysis)
