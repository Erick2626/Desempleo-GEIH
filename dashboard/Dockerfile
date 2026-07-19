FROM rocker/shiny:4.4.1

# Crear usuario que ejecuta el tablero
RUN adduser --disabled-password --gecos '' dash-user

# Instalar librerias de sistema requeridas por httr (HTTP/SSL/XML)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Definir directorio de trabajo
WORKDIR /opt/app

# Instalar dependencias (paquetes de R usados en app.R; sin xgboost, con httr/jsonlite)
ADD ./ /opt/app/
RUN R -e "install.packages(c('shiny','shinydashboard','shinycssloaders','tidyverse','plotly','DT','scales','httr','jsonlite'), repos='https://cloud.r-project.org/')"

# Hacer el run.sh ejecutable
RUN chmod +x /opt/app/run.sh
# Cambiar propiedad de la carpeta a dash-user
RUN chown -R dash-user:dash-user ./

USER dash-user
# Puerto a exponer para el tablero
EXPOSE 3838

# Comandos a ejecutar al correr el contenedor
CMD ["bash", "./run.sh"]
