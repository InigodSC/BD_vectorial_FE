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
    ```sql
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
```
#### EJECUTAR EL SCRIPT

Navega a la carpeta del archivo 'cargar_fer.py' en tu Terminal y ejecuta:
```bash
python cargar_fer.py
```

## 2. Ingesta y Migración Inicial a la nube

Esta fase cubre la migración de la base de datos local 'fer_vct' a Azure Database for PostgreSQL - Flexible Server.
Se utilizarán las herramientas nativas de PostgreSQL: `pg_dump` y `pg_restore`.

### 1. PRERREQUISITOS IMPORTANTES

* **HERRAMIENTAS**: pg_dump y pg_restore deben estar accesibles desde tu terminal. Si no lo están, usa la ruta completa del ejecutable (Ej: "C:\Program Files\PostgreSQL\18\bin\pg_dump.exe").
* **FIREWALL DE AZURE**: La IP pública de tu máquina debe estar permitida en las reglas de Firewall de tu servidor Azure.
* **CREDENCIALES**: Ten a mano el Host Name, Usuario Administrador (<AZURE_ADMIN>) y la Contraseña de tu servidor de Azure.


### 2. EXPORTAR LA BASE DE DATOS LOCAL (pg_dump)

Se crea un archivo de respaldo en la carpeta donde hemos creado.

**COMANDO DE EXPORTACIÓN:**
```bash
    <RUTA_A_PG_DUMP> -h localhost -p 5433 -U postgres -Fc -d fer_vct > fer_vct_backup.dump
```

### 3. PREPARACIÓN EN AZURE POSTGRESQL

Primero tenemos que haber desplegado el servicio de Azure Sql for PostgreSQL. En la version de postgre, podremos la misma que tenemos en local.

![alt text](image-4.png)

Luego en la seccion de `Networking` vamos a marcar las casillas que estan marcadas en la imagen y vamos a añadir nuestra ip publica para poder acceder desde nuestro equipo.

![alt text](image-5.png)

#### PERMITIR LA EXTENSIÓN EN EL PORTAL DE AZURE (**IMPORTANTE**)

Tienes que modificar un parámetro del servidor para indicarle a Azure que el usuario administrador puede habilitar esta extensión (pgvector). Este paso es obligatorio en Azure Flexible Server.

1. NAVEGA AL SERVIDOR: Ve al Portal de Azure y selecciona tu servidor PostgreSQL llamado pgvector.

2. PARÁMETROS DEL SERVIDOR: En el menú de la izquierda, busca y selecciona Parámetros del servidor (o Parameters).

3. BUSCA AZURE.EXTENSIONS: En el campo de búsqueda de parámetros, escribe azure.extensions.

4. AÑADE VECTOR: Selecciona `VECTOR` en la columna `Value`

5. HAZ CLIC EN GUARDAR (SAVE): Haz clic en Guardar en la parte superior para aplicar los cambios en la configuración del servidor.

![alt text](image-6.png)

Desde la terminal del portal de Azure vamos nos vamos a conectar a nuestro servidor y a habilitar la extensión pgvector (**IMPORTANTE**):
```bash
psql -h <AZURE_HOST_NAME> -p 5432 -U <AZURE_ADMIN> -d <NOMBRE_DB_AZURE>

CREATE EXTENSION vector;
```
![alt text](image-7.png)

### 4. IMPORTAR LA BASE DE DATOS A AZURE (pg_restore)

Utiliza pg_restore para cargar el archivo .dump en la base de datos de Azure (*esto se hace desde la terminal en local no desde Azure*).

COMANDO DE IMPORTACIÓN:

```bash
    <RUTA_A_PG_RESTORE> -h <AZURE_HOST_NAME> -p 5432 -U <AZURE_ADMIN> -d <NOMBRE_DB_AZURE> -v "<RUTA_ABSOLUTA_AL_DUMP>"
```
![alt text](image-8.png)

### 5. VERIFICACIÓN

Conéctate a la DB de Azure para confirmar la migración. Esto lo podemos hacer o desde la shell de Azure o podemos conectarnos desde PostgreSQL en local, con la ruta de Azure.

Verificar conteo de filas:
```sql
SELECT COUNT(*) FROM imagenes_fer;
```

## 3. Orquestación y Procesamiento (Azure Data Factory)

Esta fase utiliza Azure Data Factory (ADF) como la herramienta de orquestación y movimiento de datos (ELT). ADF es responsable de extraer los embeddings vectoriales desde la base de datos PostgreSQL en Azure y cargarlos en el Data Lakehouse (Azure Data Lake Storage Gen2), sirviendo como el motor de la "T" (Transformación/Movimiento) y la "L" (Carga).

### 3.1. Creación del Data Lakehouse

El Data Lakehouse se implementa utilizando **Azure Data Lake Storage Gen2 (ADLS Gen2)**, que proporciona el almacenamiento escalable y las capacidades de sistema de archivos necesarias para alojar los embeddings vectoriales en formato Parquet.

1.  **Creación de la Storage Account:**
    * En el Portal de Azure, se crea un recurso de **"Storage account"** .
    * Se selecciona la misma región y grupo de recursos utilizados para Azure PostgreSQL y Azure Data Factory para optimizar el rendimiento y la latencia.
    ![alt text](image-14.png)

2.  **Habilitación de Namespace Jerárquico (Configuración de Data Lake):**
    * **Paso Crucial:** Durante la configuración, se debe navegar a la pestaña **"Advanced"** (Opciones avanzadas).
    * En la sección **"Data Lake Storage Gen2"**, se habilita la opción **"Hierarchical namespace"** (Espacio de nombres jerárquico). Esta característica es esencial, ya que permite la gestión de directorios y archivos de manera eficiente, replicando la estructura de un sistema de archivos tradicional, lo que define el componente Data Lakehouse.
    ![alt text](image-15.png)

3.  **Organización de Contenedores:**
    * Una vez creada la cuenta, se establece una estructura lógica dentro de un contenedor (ej., 'raw' o 'landing').
    * Azure Data Factory (ADF) cargará los archivos Parquet en esta ubicación, manteniendo la estructura de carpetas definida en el Dataset de destino (ej., 'raw/embeddings/fer2013/').


### 3.2. Creación del Servicio Azure Data Factory (ADF)

1.  **Creación del Recurso ADF:** En el Portal de Azure, se crea un nuevo recurso de **Data Factory (V2)**. Se recomienda usar la misma región que Azure PostgreSQL y el Data Lakehouse para reducir la latencia.

    ![alt text](image-9.png)
    ![alt text](image-10.png)

2.  **Acceso al Studio:** Una vez desplegado, se accede al **Azure Data Factory Studio** para comenzar la configuración.

### 3.3. Configuración de Conexiones (Linked Services)

Dentro del ADF Studio, se configuran las conexiones a los almacenes de datos. Esto se realiza en la sección "Manage" (Administrar, icono del enchufe) de ADF Studio.

1.  **Linked Service para Azure PostgreSQL (Origen):**
    * **Tipo:** Azure Database for PostgreSQL.
    * **Propósito:** Permite a ADF leer los datos vectoriales de la tabla 'imagenes_fer'.
    * **Configuración en ADF:**
        * Se selecciona el tipo Azure Database for PostgreSQL.
        * Se introducen los parámetros de conexión del servidor Flexible Server: Host Name, Database name (ej., 'fer_vct'), User name (<AZURE_ADMIN>), y Password, tal como se configuraron en el Paso 2.
        * Es obligatorio realizar una prueba de conexión (Test connection) antes de crear el servicio.

    ![alt text](image-12.png)
    ![alt text](image-13.png)

2.  **Linked Service para Data Lakehouse (Destino):**
    * **Tipo:** Azure Data Lake Storage Gen2 (ADLS Gen2).
    * **Propósito:** Permite a ADF escribir los archivos Parquet resultantes en el Data Lake.
    * **Configuración en ADF:**
        * Se selecciona el tipo Azure Data Lake Storage Gen2.
        * Se elige la Cuenta de Almacenamiento (Storage account name) correspondiente a tu Data Lakehouse.
        * Se recomienda usar la Identidad Administrada (Managed Identity) para la autenticación por motivos de seguridad.
        * Se verifica la conexión y se crea el servicio.

    ![alt text](image-16.png)
    ![alt text](image-17.png)

3.  **Linked Service para Sistema de Archivos Local (Origen Local):**
    * **Tipo:** File System (Sistema de Archivos).
    * **Propósito:** Permite acceder a la carpeta de imágenes del dataset FER2013 en tu máquina local.
    * **Configuración en ADF:**
        * **Integration Runtime:** Se selecciona el **Self-hosted IR (SHIR)** previamente instalado en la máquina local.
        * **Host:** Se indica la ruta de la carpeta raíz de las imágenes (ej., 'C:\Ruta\A\Donde\Descargaste\fer2013').
        * Se proporcionan las credenciales de autenticación necesarias para que el SHIR acceda a esa ruta.
        * Se verifica la conexión y se crea el servicio.

    ![alt text](image-27.png)


### 3.4. Definición de Datasets

Los Datasets apuntan a los datos específicos dentro de las conexiones definidas. Para manejar los dos flujos de datos (embeddings desde la nube y binarios desde local) se requieren cuatro Datasets.

La configuración de estos Datasets se realiza dentro del Azure Data Factory Studio en la sección "Author" (Autor).

1.  **Dataset de Origen (DS_PgVector_ImagenesFer):**
    * **Propósito:** Apunta a la tabla 'imagenes_fer' en Azure PostgreSQL (para extraer los embeddings).
    * **Configuración en ADF:**
        * Tipo: Azure Database for PostgreSQL.
        * Linked Service: Se enlaza al servicio de PostgreSQL configurado en 3.2.1.
        * Tabla: Se selecciona la tabla 'imagenes_fer'.
    ![alt text](image-23.png)
    ![alt text](image-24.png)


2.  **Dataset de Destino (DS_Lakehouse_Embeddings):**
    * **Propósito:** Define la ubicación y el formato para los embeddings en el Data Lakehouse.
    * **Configuración en ADF:**
        * Tipo: Azure Data Lake Storage Gen2.
        * Formato: Se selecciona **Parquet** por ser columnar y optimizado para análisis.
        * Linked Service: Se enlaza al servicio de ADLS Gen2 configurado en 3.2.2.
        * Ruta de Archivo: Se especifica la ruta lógica (ej., 'raw/embeddings/fer2013/').
    ![alt text](image-20.png)
    ![alt text](image-21.png)
    ![alt text](image-22.png)

3.  **Dataset de Origen Binario (DS_Images_Source - Local):**
    * **Propósito:** Apunta a la ubicación de los archivos PNG del dataset FER2013 en el sistema de archivos local.
    * **Configuración en ADF:**
        * Tipo: **File System** (Sistema de Archivos).
        * Linked Service: Se enlaza al Linked Service del File System, el cual debe estar asociado a un **Self-hosted Integration Runtime (SHIR)** previamente instalado en la máquina local para acceder a la ruta local.
        * Formato: Se selecciona **Binary** (Binario) para tratar los archivos sin modificar su contenido.
        * Ruta: Se indica la subcarpeta donde se encuentran las imágenes dentro del Host (ej., 'train').
    ![alt text](image-25.png)
    ![alt text](image-26.png)

4.  **Dataset de Destino Binario (DS_Lakehouse_Images):**
    * **Propósito:** Define la ubicación para las imágenes originales sin alteración en el Data Lakehouse.
    * **Configuración en ADF:**
        * Tipo: Azure Data Lake Storage Gen2.
        * Formato: Se selecciona **Binary** (Binario).
        * Linked Service: Se enlaza al servicio de ADLS Gen2.

### 3.5. Creación del Pipeline de Carga (Copy Data Activity)

1.  **Crear Pipeline:** Se crea un nuevo Pipeline (ej., `PL_Migrar_Embeddings`) en ADF.
2.  **Añadir Actividad de Copia de Datos:** Se incluye una actividad de *Copy Data* para mover los datos.
    * **Configuración de Origen (Source):** Se utiliza el Dataset de origen (`DS_PgVector_ImagenesFer`).
    * **Configuración de Destino (Sink):** Se utiliza el Dataset de destino (`DS_Lakehouse_Embeddings`) configurado con el formato Parquet.
    * **Manejo de Vectores:** ADF gestiona la conversión de la columna `VECTOR(512)` de PostgreSQL en un tipo de dato compatible (como una *array* o *string* serializada) dentro del archivo Parquet.

### 3.6. Ejecución y Monitoreo

1.  **Verificación:** Se ejecuta el Pipeline en modo *Debug* para validar la transferencia de los datos vectoriales.
2.  **Programación:** Se configura un *Trigger* para automatizar la ejecución del flujo de datos (ejecución manual o programada).
3.  **Monitoreo:** Se utiliza la sección de Monitoreo de ADF para confirmar el éxito de la transferencia y el volumen de datos movido.