# Genera df_dashboard.rds: version liviana de df_modelo.rds para el tablero.
#
# El tablero (app.R) solo necesita una muestra de 50,000 filas (misma semilla
# y logica que usaba en runtime) y 13 columnas para sus graficas, filtros y
# tasas marginales. Cargar el df_modelo.rds completo (1.93M filas, ~40
# columnas) causaba un OOM kill en el contenedor (limite de 1 GB en Railway).
# Este script se ejecuta una sola vez, localmente, para producir el archivo
# liviano que el Dockerfile copia en su lugar.

library(dplyr)

df_modelo <- readRDS("df_modelo.rds")

cols_dashboard <- c(
  "ANIO", "estado_laboral",
  "sexo", "grupo_edad", "nivel_educativo", "etnia",
  "region", "zona", "tenencia",
  "estrato", "inclusion_fin_score", "servicios_basicos_score", "discapacidad"
)

set.seed(123)
df_dashboard <- df_modelo %>%
  filter(estado_laboral %in% c("Ocupado", "Desocupado")) %>%
  select(all_of(cols_dashboard)) %>%
  sample_n(min(50000, n()))

saveRDS(df_dashboard, "df_dashboard.rds")

cat("df_dashboard.rds generado:", nrow(df_dashboard), "filas x",
    ncol(df_dashboard), "columnas\n")
cat("Tamano:", round(file.size("df_dashboard.rds") / 1024^2, 2), "MB\n")
