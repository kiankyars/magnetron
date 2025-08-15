from google import genai
import os

def generate_description(topic, client):
    prompt = f"""
    EXAMPLES, DO NOT TRANSLATE THESE DIRECTLY, JUST FOR REFERENCE
    {{"prompt": "When a light metal cup falls off a counter,",
    "solution0": "it will shatter after hitting the ground.",
    "solution1": "it will bounce after hitting the ground.",
    "label": 1}}

    {{"prompt": "To make honey chipotle marinade for chicken: In a glass bowl and with a metal spoon mix together three tablespoons canola oil, two teaspoons jarred minced garlic, three tablespoons honey and two tablespoons chopped canned chipotle peppers in adobo sauce until well blended.",
    "solution0": "Pour the marinade into a ziploc bag, place your chicken pieces in the bag, squeeze out excess air and close the bag. Squish the chicken around in the marinade and leave overnight.",
    "solution1": "Pour the marinade into a paper bag, place your chicken pieces in the bag, squeeze out excess air and close the bag. Squish the chicken around in the marinade and leave overnight."
    "label": 0}}

    {{"prompt": "How do you put a stamp on a T-shirt?",
    "solution0": "You glue it on.",
    "solution1": "You iron it on.",
    "label": 1}}

    Faites svp 20 echantillions de donnés sur le sjuet: {topic}. We must also follow the guidelines: physical reasoning, common sense, and non-obviousness. Use items of variable length. Try not to include too many short items, as they may be too easy for larger models.
The two candidate solutions should be as similar as possible (e.g. differing only by one or two words, or just flipping the order of two phrases). One solution should be unambiguously correct and the other incorrect.
To ensure that the benchmark is not too "easy", the incorrect solution should not be so absurd that it is extremely obvious.
Try not to start all examples the same way."""

    return client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

if __name__ == "__main__":
    client = genai.Client()
    topics = ["la cuisine", "le sport", "l'usine", "la santé", "le transport"]
    for i, topic in enumerate(topics):
        out = generate_description(topic, client)
        with open(f"data/out{i}.json", "w", encoding="utf-8") as file:
            file.write(out.text)