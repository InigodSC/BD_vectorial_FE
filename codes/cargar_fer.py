import psycopg2
import torch
from torchvision import models, transforms
from PIL import Image
import os
import glob 


# ¡IMPORTANTE! Reemplaza esto con la ruta donde está tu carpeta 'train' del dataset FER2013
RUTA_DATASET = r"C:\Users\Íñigo\Downloads\borrar\BD_vectorial_FE\dataset\train" 

DB_PARAMS = {
    'dbname': 'fer_vct',
    'user': 'postgres',
    'password': '1234',
    'host': 'localhost',
    'port': '5433' 
}

# --- 1. CONFIGURACIÓN DEL MODELO DE EMBEDDING (ResNet18) ---
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Identity() 
model.eval() 

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), 
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 2. CONEXIÓN E INICIO ---
try:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    print("Conexión a la base de datos pgvector exitosa.")
except Exception as e:
    print(f"Error al conectar a la base de datos: {e}")
    exit()

# --- 3. PROCESAR IMÁGENES E INSERTAR VECTORES ---
total_images = 0
for emotion_folder in os.listdir(RUTA_DATASET):
    emotion_path = os.path.join(RUTA_DATASET, emotion_folder)
    
    if os.path.isdir(emotion_path):
        for img_path in glob.glob(os.path.join(emotion_path, '*.jpg')):
            
            try:
                # 1. Cargar y Transformar
                img = Image.open(img_path)
                input_tensor = transform(img)
                input_batch = input_tensor.unsqueeze(0)

                # 2. Generar el Vector
                with torch.no_grad():
                    embedding = model(input_batch).squeeze().numpy()

                # 3. Preparar para la DB
                vector_para_db = embedding.tolist() 
                relative_filepath = os.path.join(emotion_folder, os.path.basename(img_path))
                
                # 4. Insertar en pgvector
                insert_query = """
                INSERT INTO imagenes_fer (filepath, emotion, vector)
                VALUES (%s, %s, %s);
                """
                cur.execute(insert_query, (relative_filepath, emotion_folder, vector_para_db))
                total_images += 1
                
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
                continue

# --- 4. FINALIZAR TRANSACCIÓN ---
conn.commit()
cur.close()
conn.close()
print(f"--- Proceso Finalizado ---")
print(f"Total de imágenes procesadas e insertadas: {total_images}")