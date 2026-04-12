"""Built-in sample YouTube URLs for testing blink rate analysis."""

SAMPLE_VIDEOS = [
    {
        "label": "Wednesday Addams - Jenna Ortega (expect very low blink rate)",
        "url": "https://www.youtube.com/watch?v=Di0Qn1RjPzs",
        "note": "Jenna Ortega famously didn't blink while playing Wednesday",
    },
    {
        "label": "US Presidential Debate (multi-person, stress conditions)",
        "url": "https://www.youtube.com/watch?v=smkyorC5qwc",
        "note": "Good multi-person test with subjects under pressure",
    },
    {
        "label": "News Anchor - Anderson Cooper (baseline normal rate)",
        "url": "https://www.youtube.com/watch?v=PtjkQJMBl5Q",
        "note": "Professional on-camera presence, expect normal blink rate",
    },
    {
        "label": "Poker - Daniel Negreanu (trained blink control)",
        "url": "https://www.youtube.com/watch?v=VGNMDBfrb3g",
        "note": "Professional poker player, may show controlled low blink rate",
    },
    {
        "label": "Job Interview Example (normal conversational blinking)",
        "url": "https://www.youtube.com/watch?v=naIkpQ_cIt0",
        "note": "Normal conversation setting for baseline comparison",
    },
]
