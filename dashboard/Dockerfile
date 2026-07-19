FROM rocker/shiny:4.4.1

# Crear usuario que ejecuta el tablero
RUN adduser --disabled-password --gecos '' dash-user

# Instalar librerias de sistema requeridas por httr (HTTP/SSL/XML) y por las
# dependencias compiladas de tidyverse/ggplot2 (ragg, textshaping, stringi)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
    libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev libicu-dev \
    libcairo2-dev libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

# Definir directorio de trabajo
WORKDIR /opt/app

# Instalar dependencias (paquetes de R usados en app.R; sin xgboost, con httr/jsonlite).
# install.packages() no propaga error como codigo de salida: se verifica
# explicitamente que cada paquete quedo instalado, o el build falla.
ADD ./ /opt/app/
RUN R -e "pkgs <- c('shiny','shinydashboard','shinycssloaders','tidyverse','plotly','DT','scales','httr','jsonlite'); \
    install.packages(pkgs, repos='https://cloud.r-project.org/'); \
    faltantes <- setdiff(pkgs, rownames(installed.packages())); \
    if (length(faltantes) > 0) { cat('Fallo instalando:', paste(faltantes, collapse=', '), '\n'); quit(status = 1) }"

# Hacer el run.sh ejecutable
RUN chmod +x /opt/app/run.sh
# Cambiar propiedad de la carpeta a dash-user
RUN chown -R dash-user:dash-user ./

USER dash-user
# Puerto a exponer para el tablero
EXPOSE 3838

# Comandos a ejecutar al correr el contenedor
CMD ["bash", "./run.sh"]
