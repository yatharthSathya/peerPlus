import os
from getpass import getpass
from openai import OpenAI
import pandas as pd

# 🔐 Enter your API key securely
os.environ["OPENAI_API_KEY"] = ""
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

df = pd.read_csv('/root/.cache/kagglehub/datasets/dhivyeshrk/diseases-and-symptoms-dataset/versions/1/Final_Augmented_dataset_Diseases_and_Symptoms.csv')
disease_list = df['diseases'].unique()[:200]

# 💬 Chat function
def get_disease_prediction(symptoms, disease_list):
    system_prompt = (
        "You are a medical assistant. If someone asks in Spanish, answer in Spanish. Based on the symptoms provided, "
        "identify the most likely diseases (3-5) *and* suggest a the best treatment plan"
        "from the following list of possible diseases, and your own knowledge:\n\n"
        + "\n".join(disease_list)
    )


    user_prompt = f"Patient symptoms: {symptoms}\nWhat is the most likely disease from the list above?"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()

# ✅ Input + Output
symptoms = input("What are your symptoms?/ Cuales son tus sintomas?")
diagnosis = get_disease_prediction(symptoms, disease_list)
print("Predicted disease:", diagnosis)
