"""Built-in sample YouTube URLs for testing blink rate analysis.

These are public videos of well-known figures used as examples.
Replace with your own URLs for actual analysis.
"""

SAMPLE_VIDEOS = [
    {
        "label": "TV Character - No Blinking Challenge (expect very low rate)",
        "url": "https://www.youtube.com/watch?v=wykdoHz4Nfs",
        "note": "Character known for not blinking — good baseline test",
    },
    {
        "label": "Political Debate (multi-person, stress conditions)",
        "url": "https://www.youtube.com/watch?v=smkyorC5qwc",
        "note": "Good multi-person test with subjects under pressure",
    },
    {
        "label": "News Interview (baseline normal rate)",
        "url": "https://www.youtube.com/watch?v=PtjkQJMBl5Q",
        "note": "Professional on-camera presence, expect normal blink rate",
    },
    {
        "label": "Poker Player (trained blink control)",
        "url": "https://www.youtube.com/watch?v=VGNMDBfrb3g",
        "note": "Professional poker player, may show controlled low blink rate",
    },
    {
        "label": "Job Interview (normal conversational blinking)",
        "url": "https://www.youtube.com/watch?v=naIkpQ_cIt0",
        "note": "Normal conversation setting for baseline comparison",
    },
]
