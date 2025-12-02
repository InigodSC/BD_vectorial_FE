## 2. Ingesta y Migración Inicial

* **Destino Inicial:** **Azure PostgreSQL (Nube)**. Actúa como el primer punto de *staging* en la nube.

## 3. Orquestación y Preparación

* **Herramienta:** **Azure Data Factory (ADF)**.
* **Función:** Orquesta la transferencia y aplica transformaciones de datos (ELT/ETL).

## 4. Almacenamiento Centralizado (*Data Lakehouse*)

* **Destino Central:** **Data Lakehouse (Nube)**. Proporciona gobernanza, almacenamiento escalable y gestión de los datos procesados.

## 5. Destino Vectorial Optimizado

* **Destino Específico:** **Otra BD Vectorial (Nube)**. Almacén de alto rendimiento optimizado para búsquedas rápidas de similitud (*vector search*).

## 6. Consumo y Análisis Avanzado

* **Plataforma de Consumo:** **Microsoft AI Foundry (Nube)**. Accede a la BD Vectorial para el entrenamiento de modelos, RAG y la entrega de aplicaciones de IA Generativa.