FROM python:3.12-slim

# Crear usuario que ejecuta la app
RUN adduser --disabled-password --gecos '' api-user

# Instalar compiladores C/C++ y build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Definir directorio de trabajo
WORKDIR /opt/desempleo-api

# Instalar dependencias (incluye el .whl del modelo referenciado en requirements.txt)
ADD ./ /opt/desempleo-api/
RUN pip install --upgrade pip
RUN pip install -r /opt/desempleo-api/requirements.txt

# Hacer el run.sh ejecutable
RUN chmod +x /opt/desempleo-api/run.sh
# Cambiar propiedad de la carpeta a api-user
RUN chown -R api-user:api-user ./

USER api-user
# Puerto a exponer para la api
EXPOSE 8001

# Comandos a ejecutar al correr el contenedor
CMD ["bash", "./run.sh"]
