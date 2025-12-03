# ETL con PostgreSQL

Este documento describe el flujo de datos para la Extracción, Transformación y Carga (ETL) de datos vectoriales, desde un entorno local basado en PostgreSQL hasta la plataforma de análisis y consumo de Inteligencia Artificial en la nube de Azure.



## 1 CREACIÓN DE BD Y CARGA DE DATOS EN POSTGRESQL

El objetivo es crear una Base de Datos Vectorial (pgvector) y cargar las imágenes del dataset FER2013.
Para ello, utilizaremos **Docker**, que tiene un contenedor con el servidor **PostgreSQL** y la extensión **pgvector** preinstalados.


### 1.1. CREACIÓN DE LA BASE DE DATOS Y LA TABLA

#### DESCARGA Y EJECUCIÓN DEL CONTENEDOR DOCKER (Pgvector)

Usaremos la imagen ankane/pgvector.

1.  **LIMPIEZA (Opcional):** Si tienes un contenedor previo llamado 'mi_postgres_pgvector' o el puerto 5433 en uso, elimínalo:

    ```powershell
        docker stop mi_postgres_pgvector
        docker rm mi_postgres_pgvector
    ```

2.  **EJECUTAR EL CONTENEDOR:** Ejecuta el siguiente comando en una sola línea en tu Terminal (PowerShell). Usamos el puerto **5433** en el host para evitar conflictos.

    ```powershell
        docker run -e POSTGRES_USER=<usuario> -e POSTGRES_PASSWORD=<contraseña> -e POSTGRES_DB=<nombre de la bd> --name <nombre del contenedor> -p 5433:5432 -d ankane/pgvector
    ```
    Esto tambien se puede hacer desde Docker Desktop. Para ello buscamos la imagen y luego iniciamos le contenedor con la misma configuracion que por comando.
    ![alt text](image.png)
    ![alt text](image-1.png)
    ![alt text](image-2.png)

**DETALLES DE CONFIGURACIÓN**
* Puerto de tu PC (Host): `5433` (u otro en el que no este alojado ya algun servidor en postgreSql)
* Usuario/Contraseña: \<Tu usuario> / \<Tu contraseña>
* Base de datos creada: `fer_vct`

#### HABILITACIÓN DE LA EXTENSIÓN PGVECTOR

Una vez que el contenedor está corriendo, la extensión debe activarse en la base de datos:

1.  **CONECTAR A LA CONSOLA (Terminal):**
    ```bash
        docker exec -it <nombre del contenedor> psql -U <usuario> -d <nombre de la bd>
    ```
2.  **EJECUTAR SQL:** En 'fer_vct=#', escribe:
    ```bash
        CREATE EXTENSION vector;
    ```
    (Escribe \q y Enter para salir)

#### CONEXIÓN DESDE PGADMIN 4 Y CREACIÓN DE LA TABLA

1.  **CONEXIÓN EN PGADMIN:** Crea un nuevo servidor con la siguiente configuración:
    * Host name/address: `localhost`
    * Port (Puerto): `5433` (u otro en el que no este alojado ya algun servidor en postgreSql)
    * Username: \<Tu usuario>
    * Password: \<Tu contraseña>

2.  **CREACIÓN DE LA TABLA:** Una vez conectado a la base de datos 'fer_vct', ejecuta este SQL en la Query Tool. Usaremos 512 dimensiones, un tamaño común para embeddings generados por modelos como ResNet.

``` sql
CREATE TABLE imagenes_fer (
    id bigserial PRIMARY KEY, 
    filepath VARCHAR(255) NOT NULL, -- Ruta o nombre del archivo original
    emotion VARCHAR(50) NOT NULL, -- Etiqueta de emoción (ej: Feliz, Triste)
    vector VECTOR(512) -- Columna que almacenará el vector numérico
);
```
![alt text](image-3.png)


### 1.2. CARGA DE DATOS VECTORIALES (EMBEDDINGS)

#### EXPLICACIÓN DEL PROCESO DE CARGA

La base de datos vectorial solo almacena **vectores numéricos**, no imágenes. Por lo tanto, el proceso requiere un script de Python que:

1.  Utilice un **Modelo de Deep Learning pre-entrenado** (como ResNet) para extraer las características de cada imagen.
2.  Convierta esas características en una **lista de 512 números (el embedding)**.
3.  Utilice la librería **psycopg2** para enviar ese vector a la columna VECTOR(512) de PostgreSQL.

#### DATASET FER2013

El código está adaptado para procesar este dataset de expresiones faciales:

Enlace al Dataset: https://www.kaggle.com/datasets/msambare/fer2013/data

#### SCRIPT DE PYTHON (cargar_fer.py)

1.  **INSTALAR LIBRERÍAS:**
    ``` bash
        pip install psycopg2-binary numpy torch torchvision
    ```
2.  **SCRIPT:** Guarda el siguiente código como 'cargar_fer.py'. **¡IMPORTANTE!** Reemplaza la variable RUTA_DATASET con la ubicación real de tu carpeta 'train'.

```python
import psycopg2
import torch
from torchvision import models, transforms
from PIL import Image
import os
import glob 


# ¡IMPORTANTE! Reemplaza esto con la ruta donde está tu carpeta 'train' del dataset FER2013
RUTA_DATASET = 'C:/Ruta/A/Donde/Descargaste/fer2013/train' 

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
        for img_path in glob.glob(os.path.join(emotion_path, '*.png')):
            
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
                INSERT INTO fer2013_embeddings (filepath, emotion, vector_embedding)
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
```
#### EJECUTAR EL SCRIPT

Navega a la carpeta del archivo 'cargar_fer.py' en tu Terminal y ejecuta:
```bash
python cargar_fer.py
```

