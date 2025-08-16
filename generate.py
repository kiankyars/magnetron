from google import genai

def generate_description(topic, client, i):
    prompt = f"""
    EXEMPLES EN FRANÇAIS:
    {{"prompt": "Quand une tasse métallique légère tombe du comptoir,",
    "solution0": "elle se brise en touchant le sol.",
    "solution1": "elle rebondit en touchant le sol.",
    "label": 1}}

    {{"prompt": "Pour faire une marinade miel-chipotle pour le poulet: Dans un bol en verre avec une cuillère métallique, mélangez trois cuillères à soupe d'huile de colza, deux cuillères à café d'ail haché en pot, trois cuillères à soupe de miel et deux cuillères à soupe de piments chipotle hachés en conserve dans la sauce adobo jusqu'à bien mélangé.",
    "solution0": "Versez la marinade dans un sac ziploc, placez vos morceaux de poulet dans le sac, pressez l'air excédentaire et fermez le sac. Malaxez le poulet dans la marinade et laissez reposer toute la nuit.",
    "solution1": "Versez la marinade dans un sac en papier, placez vos morceaux de poulet dans le sac, pressez l'air excédentaire et fermez le sac. Malaxez le poulet dans la marinade et laissez reposer toute la nuit.",
    "label": 0}}

    {{"prompt": "Comment mettre un tampon sur un T-shirt?",
    "solution0": "Vous le collez.",
    "solution1": "Vous le repassez.",
    "label": 1}}

    {{"prompt": "Pour améliorer ses performances en course à pied,",
    "solution0": "il faut s'entraîner régulièrement et progressivement.",
    "solution1": "il faut s'entraîner intensément et rapidement.",
    "label": 0}}

    {{"prompt": "En cas de panne de voiture sur l'autoroute,",
    "solution0": "garez-vous sur la bande d'arrêt d'urgence.",
    "solution1": "garez-vous sur la voie de gauche.",
    "label": 0}}

    Créez 20 échantillons de données sur le sujet: **{topic}**. Chaque phrase sera TOUJOURS d'une longeur d'environ: **{10 + i*5}** mots.
    Les deux solutions candidates doivent TOUJOURS différer d'un maximum d'un ou deux mots seulement. 
    Une solution doit être correcte, l'autre plausible, NE FAITES PAS D'ÉCHANTILLIONS OÙ LA RÈPONSE EST TROP ÉVIDENTE. 
    Utilisez le raisonnement physique et le bon sens."""
    return client.models.generate_content(model="gemini-2.5-pro", contents=prompt)

if __name__ == "__main__":
    client = genai.Client()
    topics = ["la cuisine", "le sport", "l'usine", "la santé", "le transport", "l'éducation", "la technologie", "l'environnement", "les arts", "le commerce"]
    for i, topic in enumerate(topics):
        out = generate_description(topic, client, i)
        with open(f"data/out{i}.json", "w", encoding="utf-8") as file:
            file.write(out.text)