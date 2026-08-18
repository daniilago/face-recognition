import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
from deepface import DeepFace

path = "images"
cache_file = "embeddings_cache.json"

def generate_cache():
    database = []

    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)

        try:
            embbeding_obj = DeepFace.represent(
                img_path=filepath,
                model_name='ArcFace',
                detector_backend='retinaface',
                enforce_detection=False
            )

            name, _ = os.path.splitext(filename)

            database.append({
                'name': name,
                'embedding': embbeding_obj[0]['embedding']
            })

            print(f"Processado: {name}")

        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")

    with open(cache_file, 'w') as f:
        json.dump(database, f)

    print(f"\nCache salvo em {cache_file} com {len(database)} pessoas")

if __name__ == "__main__":
    generate_cache()