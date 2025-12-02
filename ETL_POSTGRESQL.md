# ETL con PostgreSQL

Este documento describe el flujo de datos para la Extracción, Transformación y Carga (ETL) de datos vectoriales, desde un entorno local basado en PostgreSQL hasta la plataforma de análisis y consumo de Inteligencia Artificial en la nube de Azure.

## 1. Creación de BD y Carga de Datos en PostgreSQL

### 1.1. Instalación de la Extensión pgvector (Windows)

La instalación de `pgvector` requiere la **compilación** del código fuente para que funcione con tu servidor PostgreSQL en Windows.

#### Requisitos Dobles: Compilador (MSVC) y Utilidad `make` (MSYS2)

* **1. Compilador de C++ (MSVC):** Necesario para el comando **`cl`**. Se obtiene instalando las **Visual Studio Build Tools** y marcando **"Desarrollo para el escritorio con C++"**.
* **2. Utilidad `make.exe`:** Necesaria para ejecutar el `Makefile`. Se obtiene a través de **MSYS2** (`pacman -S make`).

#### Clonar y Configurar el Entorno (Solución Final a 'cl' y 'pg_config')

Debido a que `make` y `cl` usan entornos diferentes, debemos configurar el terminal MSYS2 (donde `make` funciona sin errores de DLL) para que conozca las rutas de Visual Studio.

1.  **Clonar el Repositorio (Recomendado en PowerShell):**
    ```bash
    git clone https://github.com/pgvector/pgvector.git
    ```

2.  **Abrir el Terminal MSYS2:** Abre el programa llamado **MSYS2 MinGW 64-bit**.

3.  **Navegar y Definir Rutas de PostgreSQL (Usando Ruta Corta):** La ruta corta (`PROGRA~1`) es crucial para evitar errores de espacios en MSYS2.

    ```bash
    cd /c/pgvector 
    
    export PGROOT="/c/PROGRA~1/PostgreSQL/18"
    export PG_CONFIG="/c/PROGRA~1/PostgreSQL/18/bin/pg_config.exe"
    ```

4.  **Añadir el Compilador al PATH (Solución al error 'cl'):** Debes añadir la ruta de tu compilador MSVC al PATH de MSYS2.

    * **Paso Previo:** Encuentra la ruta exacta del `cl.exe` de tu instalación de Visual Studio (ej: `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64`).

    REEMPLAZA la ruta con la que encontraste en tu sistema:
    ```bash 
    export PATH=$PATH:"/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.37.32822/bin/Hostx64/x64"
    ```

#### Compilación e Instalación Final

Ejecuta los comandos para compilar (`make`) e instalar (`make install`).

```bash
make
make install
```

### 1.2. Habilitación de pgvector en PgAdmin 4

Una vez que los archivos binarios de la extensión han sido copiados al servidor (tras ejecutar `make install`), puedes activarla en la base de datos específica donde deseas almacenar y consultar los vectores.

1.  **Abrir PgAdmin 4:** Conéctate a tu servidor PostgreSQL.
2.  **Abrir Query Tool:** Selecciona la base de datos de destino (o crea una nueva) y abre la herramienta de consulta (**Query Tool**).
3.  **Ejecutar el Comando SQL:** Ejecuta el siguiente comando para habilitar la extensión `vector`:

    ```sql
    CREATE EXTENSION vector;
    ```

4.  **Verificación:** Para confirmar la instalación, navega en el árbol de objetos de PgAdmin:
    * Databases $\rightarrow$ Tu\_Base\_de\_Datos $\rightarrow$ Schemas $\rightarrow$ public $\rightarrow$ **Extensions**.
    * La extensión **`vector`** deberá aparecer en la lista, indicando que está lista para ser utilizada.


