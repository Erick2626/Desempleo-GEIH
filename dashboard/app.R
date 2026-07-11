library(shiny)
library(shinydashboard)
library(shinycssloaders)
library(tidyverse)
library(xgboost)
library(plotly)
library(DT)
library(scales)

# =============================================================================
# CARGA DE MODELO Y DATOS
# =============================================================================
cat("Cargando modelo y datos...\n")

setwd("C:/Users/corys/OneDrive/Desktop/MIIA/Semestres/III/Despliegue/Desempleo-GEIH/dashboard")

if (!file.exists("df_modelo.rds")) {
  stop("No se encuentra df_modelo.rds. Copia el archivo a la carpeta del app.")
}
if (!file.exists("modelo_xgb.model")) {
  stop("No se encuentra modelo_xgb.model. Copia el archivo a la carpeta del app.")
}

modelo_xgb <- xgb.load("modelo_xgb.model")
df_modelo <- readRDS("df_modelo.rds")

# =============================================================================
# CONSTANTES
# =============================================================================
THRESHOLD_OPTIMO <- 0.493
TASA_NACIONAL <- 11.3

# =============================================================================
# MUESTRA PARA EDA
# =============================================================================
set.seed(123)
df_ft <- df_modelo %>%
  filter(estado_laboral %in% c("Ocupado", "Desocupado")) %>%
  sample_n(min(50000, n()))

# =============================================================================
# VARIABLES PREDICTORAS
# =============================================================================
vars_x <- c("sexo", "grupo_edad", "nivel_educativo", "etnia",
            "discapacidad", "jefe_hogar", "mayor_18",
            "region", "zona",
            "estrato", "hacinamiento", "servicios_basicos_score",
            "tenencia", "n_menores_15", "n_mayores_65", "razon_dependencia",
            "inclusion_fin_score", "sin_producto_fin",
            "transferencias_gov", "recibe_remesas")

# =============================================================================
# MATRIZ DE REFERENCIA
# =============================================================================
set.seed(42)
df_template <- df_modelo %>%
  filter(estado_laboral %in% c("Ocupado", "Desocupado")) %>%
  select(all_of(c(vars_x, "desempleado"))) %>%
  mutate(y = factor(ifelse(desempleado == 1, "D", "O"), levels = c("D", "O"))) %>%
  select(-desempleado) %>%
  drop_na() %>%
  sample_n(1000)

mat_ref <- model.matrix(y ~ . - 1, data = df_template)
cols_modelo <- colnames(mat_ref)

# =============================================================================
# OPCIONES DE SELECTORES
# =============================================================================
opciones <- list(
  sexo = levels(df_modelo$sexo),
  grupo_edad = c("15-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"),
  nivel_educativo = levels(df_modelo$nivel_educativo),
  etnia = levels(df_modelo$etnia),
  region = levels(df_modelo$region),
  zona = levels(df_modelo$zona),
  tenencia = levels(df_modelo$tenencia)
)

# =============================================================================
# TASAS MARGINALES
# =============================================================================
cat("Calculando tasas marginales...\n")
tasas_marginales <- list()

for (var in c("sexo", "grupo_edad", "nivel_educativo", "etnia", "region", "zona", "tenencia")) {
  tasas_marginales[[var]] <- df_ft %>%
    group_by(across(all_of(var))) %>%
    summarise(tasa = round(100 * mean(estado_laboral == "Desocupado"), 2), .groups = "drop") %>%
    deframe()
}

tasas_marginales$estrato <- df_ft %>%
  filter(estrato %in% 1:6) %>%
  group_by(estrato) %>%
  summarise(tasa = round(100 * mean(estado_laboral == "Desocupado"), 2), .groups = "drop") %>%
  deframe()

tasas_marginales$inclusion_fin_score <- df_ft %>%
  group_by(inclusion_fin_score) %>%
  summarise(tasa = round(100 * mean(estado_laboral == "Desocupado"), 2), .groups = "drop") %>%
  deframe()

tasas_marginales$servicios_basicos_score <- df_ft %>%
  group_by(servicios_basicos_score) %>%
  summarise(tasa = round(100 * mean(estado_laboral == "Desocupado"), 2), .groups = "drop") %>%
  deframe()

tasas_marginales$discapacidad <- df_ft %>%
  group_by(discapacidad) %>%
  summarise(tasa = round(100 * mean(estado_laboral == "Desocupado"), 2), .groups = "drop") %>%
  deframe()

# =============================================================================
# PERFILES PRECONFIGURADOS
# =============================================================================
PERFILES_PRESET <- list(
  "profesional_bogota" = list(
    label = "Profesional Bogotá", color = "primary",
    descripcion = "Hombre, 35-44, universitario, Bogotá, estrato 4",
    sexo = "Masculino", grupo_edad = "35-44", nivel_educativo = "Universitaria",
    etnia = "Ningun grupo etnico", discapacidad = FALSE, jefe_hogar = TRUE,
    region = "Bogota DC", zona = "Cabecera", estrato = 4, tenencia = "Propia",
    servicios = 5, hacinamiento = 1, inclusion = 4, sin_producto = FALSE,
    n_menores = 1, n_mayores = 0, transferencias = 0, remesas = FALSE
  ),
  "joven_caribe" = list(
    label = "Joven trabajador Caribe", color = "warning",
    descripcion = "Hombre, 18-24, técnico, Atlántico, estrato 2",
    sexo = "Masculino", grupo_edad = "18-24",
    nivel_educativo = "Tecnica/Tecnologica/Normalista", etnia = "Ningun grupo etnico",
    discapacidad = FALSE, jefe_hogar = FALSE, region = "Caribe", zona = "Cabecera",
    estrato = 2, tenencia = "Arrendada", servicios = 4, hacinamiento = 2,
    inclusion = 2, sin_producto = FALSE, n_menores = 0, n_mayores = 1,
    transferencias = 0, remesas = FALSE
  ),
  "mujer_joven_pacifico" = list(
    label = "Mujer joven Pacífico", color = "danger",
    descripcion = "Mujer, 18-24, media, Pacífica, estrato 1",
    sexo = "Femenino", grupo_edad = "18-24",
    nivel_educativo = "Media (academica/tecnica)", etnia = "Afrocolombiano",
    discapacidad = FALSE, jefe_hogar = FALSE, region = "Pacifica", zona = "Cabecera",
    estrato = 1, tenencia = "Arrendada", servicios = 3, hacinamiento = 3,
    inclusion = 0, sin_producto = TRUE, n_menores = 2, n_mayores = 0,
    transferencias = 1, remesas = FALSE
  ),
  "adulto_mayor_rural" = list(
    label = "Adulto mayor rural", color = "purple",
    descripcion = "Hombre, 65+, primaria, Andina rural, estrato 1",
    sexo = "Masculino", grupo_edad = "65+", nivel_educativo = "Basica primaria",
    etnia = "Ningun grupo etnico", discapacidad = TRUE, jefe_hogar = TRUE,
    region = "Andina", zona = "Resto (centro poblado y rural disperso)",
    estrato = 1, tenencia = "Propia", servicios = 2, hacinamiento = 2,
    inclusion = 1, sin_producto = FALSE, n_menores = 0, n_mayores = 1,
    transferencias = 1, remesas = FALSE
  )
)

cat("App lista. df_ft:", nrow(df_ft), "filas | cols matriz:", length(cols_modelo), "\n")

# =============================================================================
# PALETAS Y TEMA
# =============================================================================
PAL_ESTADO <- c("Ocupado" = "#A8C5E0", "Desocupado" = "#E89B97")
PAL_SEXO <- c("Masculino" = "#A8C5E0", "Femenino" = "#E5B3D1")

tema_app <- theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(color = "gray40"),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

# =============================================================================
# UI
# =============================================================================
ui <- dashboardPage(
  skin = "blue",
  
  dashboardHeader(
    title = span(icon("chart-line"), " Predictor Desempleo Colombia"),
    titleWidth = 300
  ),
  
  dashboardSidebar(
    width = 230,
    sidebarMenu(
      id = "menu",
      menuItem("Panorama general", tabName = "overview", icon = icon("globe-americas")),
      menuItem("Predicción individual", tabName = "predict", icon = icon("user-check"))
    ),
    hr()
    
  ),
  
  dashboardBody(
    tags$head(tags$style(HTML("
            .content-wrapper, .right-side { background-color: #F7F9FB; }

            /* Tarjeta de probabilidad */
            .caja-probabilidad { padding: 25px 20px; border-radius: 12px; text-align: center; margin: 10px 0; }
            .caja-probabilidad .num { font-size: 72px; font-weight: 800; line-height: 1; color: #1a2634; }
            .caja-probabilidad .label { font-size: 13px; color: #6B7B85; margin-top: 6px; letter-spacing: 0.5px; text-transform: uppercase; }
            .caja-probabilidad.bajo  { background: #DCEED1; }
            .caja-probabilidad.medio { background: #FBE8C9; }
            .caja-probabilidad.alto  { background: #F8CDC9; }

            /* Barra de tiempo */
            .barra-tiempo { width: 100%; height: 10px; border-radius: 8px; background: #e9ecef; margin: 12px 0; overflow: hidden; }
            .barra-tiempo .relleno { height: 100%; border-radius: 8px; transition: width 0.3s; }
            .barra-tiempo .relleno.bajo  { background: #74C476; }
            .barra-tiempo .relleno.medio { background: #F6D6AC; }
            .barra-tiempo .relleno.alto  { background: #E89B97; }

            .etiqueta-riesgo { font-size: 14px; font-weight: 600; margin-top: 4px; }

            /* Perfiles presets */
            .preset-btn { width: 100%; padding: 10px; margin: 4px 0; border-radius: 6px; border: none;
                          cursor: pointer; font-weight: 600; color: #1a2634; text-align: left;
                          transition: transform 0.1s; font-size: 13px; }
            .preset-btn:hover { transform: translateY(-2px); opacity: 0.85; }
            .preset-btn small { display: block; font-weight: normal; opacity: 0.75; margin-top: 2px; font-size: 11px; }
            .bg-primary { background: #A8C5E0 !important; }
            .bg-warning { background: #F6D6AC !important; }
            .bg-danger  { background: #F1B7B5 !important; }
            .bg-purple  { background: #D4C5E2 !important; }

            /* SHAP */
            .shap-item { display: flex; align-items: center; padding: 6px 0; border-bottom: 1px solid #f0f0f0; }
            .shap-item .shap-nombre { flex: 1; font-size: 13px; color: #2F4858; }
            .shap-item .shap-barra { height: 8px; border-radius: 4px; margin: 0 10px; }
            .shap-item .shap-valor { font-size: 12px; font-weight: 600; min-width: 40px; text-align: right; }
            .shap-positivo { background: #E89B97; }
            .shap-negativo { background: #A8C5E0; }

            .box.box-solid.box-primary > .box-header { background: #A8C5E0 !important; color: #1a2634 !important; }
            .box.box-solid.box-primary { border-color: #A8C5E0 !important; }
            .box.box-solid.box-success > .box-header { background: #B5D8B5 !important; color: #1a2634 !important; }
            .box.box-solid.box-success { border-color: #B5D8B5 !important; }
            .box.box-solid.box-info > .box-header { background: #C1D9DC !important; color: #1a2634 !important; }
            .box.box-solid.box-info { border-color: #C1D9DC !important; }
            .box.box-solid.box-warning > .box-header { background: #F6D6AC !important; color: #1a2634 !important; }
            .box.box-solid.box-warning { border-color: #F6D6AC !important; }
            .box.box-solid.box-danger > .box-header { background: #F1B7B5 !important; color: #1a2634 !important; }
            .box.box-solid.box-danger { border-color: #F1B7B5 !important; }
        "))),
    
    tabItems(
      # =================================================================
      # PESTA?A 1: PANORAMA GENERAL
      # =================================================================
      tabItem(tabName = "overview",
              fluidRow(
                box(width = 12, status = "primary", solidHeader = TRUE,
                    title = "Panorama general",
                    fluidRow(
                      column(3,
                             selectInput("filtro_anio", "Año:",
                                         choices = c("Todos" = "todos", sort(unique(df_ft$ANIO))),
                                         selected = "todos")
                      ),
                      column(3,
                             selectInput("filtro_departamento", "Departamento:",
                                         choices = c("Todos" = "todos", sort(unique(df_ft$region))),
                                         selected = "todos")
                      ),
                      column(3,
                             selectInput("filtro_edad", "Grupo de edad:",
                                         choices = c("Todos" = "todos", opciones$grupo_edad),
                                         selected = "todos")
                      ),
                      column(3,
                             div(style = "margin-top: 25px;",
                                 tags$span(style = "color: #6B7B85; font-size: 12px;",
                                           "Tasa nacional: ", strong(TASA_NACIONAL, "%"))
                             )
                      )
                    )
                )
              ),
              
              # Mapa de calor
              fluidRow(
                box(width = 12, status = "primary", solidHeader = TRUE,
                    title = "Mapa de calor departamental",
                    subtitle = "Tasa de desempleo por departamento",
                    withSpinner(plotlyOutput("heatmap", height = "420px"), type = 6)
                )
              ),
              
              # Pir?mide poblacional + Serie de tiempo
              fluidRow(
                box(width = 6, status = "primary", solidHeader = TRUE,
                    title = "Pirámide poblacional",
                    subtitle = "Segmentada por condición de actividad",
                    withSpinner(plotlyOutput("piramide", height = "380px"), type = 6)
                ),
                box(width = 6, status = "primary", solidHeader = TRUE,
                    title = "Serie de tiempo mensual 2023-2025",
                    subtitle = "Tasa de desempleo por sexo y grupo de edad",
                    withSpinner(plotlyOutput("serie_tiempo", height = "380px"), type = 6)
                )
              )
      ),
      
      # =================================================================
      # PESTA?A 2: PREDICCI?N INDIVIDUAL
      # =================================================================
      tabItem(tabName = "predict",
              fluidRow(
                box(width = 12, status = "success", solidHeader = TRUE,
                    title = "Predicción individual",
                    p(style = "font-size: 14px;",
                      "Ingresa el perfil sociodemográfico o usa un perfil preconfigurado.")
                )
              ),
              
              # Perfiles preconfigurados
              fluidRow(
                box(width = 12, status = "warning", solidHeader = TRUE,
                    title = "Perfiles preconfigurados",
                    fluidRow(
                      column(3, actionButton("preset_1", class = "preset-btn bg-primary",
                                             HTML(paste0("<b>", PERFILES_PRESET$profesional_bogota$label, "</b>",
                                                         "<small>", PERFILES_PRESET$profesional_bogota$descripcion, "</small>")))),
                      column(3, actionButton("preset_2", class = "preset-btn bg-warning",
                                             HTML(paste0("<b>", PERFILES_PRESET$joven_caribe$label, "</b>",
                                                         "<small>", PERFILES_PRESET$joven_caribe$descripcion, "</small>")))),
                      column(3, actionButton("preset_3", class = "preset-btn bg-danger",
                                             HTML(paste0("<b>", PERFILES_PRESET$mujer_joven_pacifico$label, "</b>",
                                                         "<small>", PERFILES_PRESET$mujer_joven_pacifico$descripcion, "</small>")))),
                      column(3, actionButton("preset_4", class = "preset-btn bg-purple",
                                             HTML(paste0("<b>", PERFILES_PRESET$adulto_mayor_rural$label, "</b>",
                                                         "<small>", PERFILES_PRESET$adulto_mayor_rural$descripcion, "</small>"))))
                    )
                )
              ),
              
              # Formulario
              fluidRow(
                column(4,
                       box(width = NULL, title = "Sexo y edad", status = "primary", solidHeader = TRUE,
                           selectInput("in_sexo", "Sexo:", choices = c("Femenino", "Masculino"), selected = "Femenino"),
                           selectInput("in_grupo_edad", "Grupo de edad:", choices = opciones$grupo_edad, selected = "25-34"),
                           selectInput("in_nivel_educativo", "Nivel educativo:", choices = opciones$nivel_educativo,
                                       selected = "Media (academica/tecnica)")
                       ),
                       box(width = NULL, title = "Características adicionales", status = "primary", solidHeader = TRUE,
                           selectInput("in_etnia", "Etnia:", choices = opciones$etnia, selected = "Ningun grupo etnico"),
                           checkboxInput("in_discapacidad", "Tiene discapacidad", FALSE),
                           checkboxInput("in_jefe_hogar", "Es jefe(a) de hogar", FALSE)
                       )
                ),
                column(4,
                       box(width = NULL, title = "Ubicación y vivienda", status = "primary", solidHeader = TRUE,
                           selectInput("in_region", "Departamento:", choices = opciones$region, selected = "Andina"),
                           selectInput("in_zona", "Zona:", choices = opciones$zona, selected = "Cabecera"),
                           numericInput("in_estrato", "Estrato (1-6):", value = 2, min = 1, max = 6, step = 1),
                           selectInput("in_tenencia", "Tenencia vivienda:", choices = opciones$tenencia, selected = "Arrendada"),
                           sliderInput("in_servicios", "Servicios básicos (0-5):", value = 4, min = 0, max = 5, step = 1),
                           sliderInput("in_hacinamiento", "Hacinamiento (personas/cuarto):", value = 2, min = 0.5, max = 8, step = 0.5)
                       )
                ),
                column(4,
                       box(width = NULL, title = "Hogar e inclusión", status = "primary", solidHeader = TRUE,
                           sliderInput("in_inclusion", "Inclusión financiera (0-4):", value = 2, min = 0, max = 4, step = 1),
                           checkboxInput("in_sin_producto", "Sin productos financieros", FALSE),
                           sliderInput("in_n_menores", "Menores de 15 en hogar:", value = 1, min = 0, max = 10, step = 1),
                           sliderInput("in_n_mayores", "Adultos 65+ en hogar:", value = 0, min = 0, max = 5, step = 1),
                           sliderInput("in_transferencias", "Transferencias gov. (0-3):", value = 0, min = 0, max = 3, step = 1),
                           checkboxInput("in_remesas", "Recibe remesas", FALSE),
                           hr(),
                           actionButton("btn_predecir", "Calcular probabilidad",
                                        class = "btn-success btn-block",
                                        style = "font-size: 16px; font-weight: bold;")
                       )
                )
              ),
              
              # Resultados
              fluidRow(
                box(width = 5, title = "Probabilidad estimada de desempleo",
                    status = "success", solidHeader = TRUE,
                    withSpinner(uiOutput("caja_probabilidad"), type = 6)
                ),
                box(width = 7, title = "Importancia de variables (SHAP)",
                    status = "info", solidHeader = TRUE,
                    withSpinner(uiOutput("shap_importancia"), type = 6)
                )
              )
      )
    )
  )
)

# =============================================================================
# SERVER
# =============================================================================
server <- function(input, output, session) {
  
  # -------------------------------------------------------------------------
  # Cargar perfil preconfigurado
  # -------------------------------------------------------------------------
  cargar_perfil <- function(perfil) {
    updateSelectInput(session, "in_sexo", selected = perfil$sexo)
    updateSelectInput(session, "in_grupo_edad", selected = perfil$grupo_edad)
    updateSelectInput(session, "in_nivel_educativo", selected = perfil$nivel_educativo)
    updateSelectInput(session, "in_etnia", selected = perfil$etnia)
    updateCheckboxInput(session, "in_discapacidad", value = perfil$discapacidad)
    updateCheckboxInput(session, "in_jefe_hogar", value = perfil$jefe_hogar)
    updateSelectInput(session, "in_region", selected = perfil$region)
    updateSelectInput(session, "in_zona", selected = perfil$zona)
    updateNumericInput(session, "in_estrato", value = perfil$estrato)
    updateSelectInput(session, "in_tenencia", selected = perfil$tenencia)
    updateSliderInput(session, "in_servicios", value = perfil$servicios)
    updateSliderInput(session, "in_hacinamiento", value = perfil$hacinamiento)
    updateSliderInput(session, "in_inclusion", value = perfil$inclusion)
    updateCheckboxInput(session, "in_sin_producto", value = perfil$sin_producto)
    updateSliderInput(session, "in_n_menores", value = perfil$n_menores)
    updateSliderInput(session, "in_n_mayores", value = perfil$n_mayores)
    updateSliderInput(session, "in_transferencias", value = perfil$transferencias)
    updateCheckboxInput(session, "in_remesas", value = perfil$remesas)
  }
  
  observeEvent(input$preset_1, { cargar_perfil(PERFILES_PRESET$profesional_bogota) })
  observeEvent(input$preset_2, { cargar_perfil(PERFILES_PRESET$joven_caribe) })
  observeEvent(input$preset_3, { cargar_perfil(PERFILES_PRESET$mujer_joven_pacifico) })
  observeEvent(input$preset_4, { cargar_perfil(PERFILES_PRESET$adulto_mayor_rural) })
  
  # -------------------------------------------------------------------------
  # Data reactiva para filtros
  # -------------------------------------------------------------------------
  df_filtrado <- reactive({
    d <- df_ft
    if (input$filtro_anio != "todos") {
      d <- d %>% filter(ANIO == as.integer(input$filtro_anio))
    }
    if (input$filtro_departamento != "todos") {
      d <- d %>% filter(region == input$filtro_departamento)
    }
    if (input$filtro_edad != "todos") {
      d <- d %>% filter(grupo_edad == input$filtro_edad)
    }
    d
  })
  
  # -------------------------------------------------------------------------
  # Heatmap (usando regiones como proxy de departamentos)
  # -------------------------------------------------------------------------
  output$heatmap <- renderPlotly({
    df_heat <- df_filtrado() %>%
      filter(!is.na(region), region != "No informa") %>%
      group_by(region) %>%
      summarise(
        tasa = round(100 * mean(estado_laboral == "Desocupado"), 2),
        n = n(),
        .groups = "drop"
      ) %>%
      arrange(desc(tasa))
    
    df_heat <- df_heat %>%
      mutate(region = factor(region, levels = rev(unique(region))),
             color = ifelse(tasa > TASA_NACIONAL * 1.3, "#D7191C",
                            ifelse(tasa > TASA_NACIONAL, "#FDAE61", "#74C476")))
    
    p <- ggplot(df_heat, aes(x = 1, y = region, fill = tasa,
                             text = paste0("<b>", region, "</b><br>Tasa: ", tasa, "%<br>N: ", format(n, big.mark = ",")))) +
      geom_tile(color = "white", linewidth = 0.5) +
      geom_text(aes(label = paste0(tasa, "%")), size = 3.5, color = "#1a2634", fontface = "bold") +
      scale_fill_gradient(low = "#74C3D0", high = "#D7191C", name = "Tasa %") +
      scale_x_continuous(expand = c(0, 0)) +
      labs(x = NULL, y = NULL, title = paste0("Tasa de desempleo por departamento")) +
      tema_app +
      theme(
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.y = element_text(size = 10),
        panel.grid = element_blank(),
        legend.position = "none"
      )
    
    ggplotly(p, tooltip = "text", height = 400) %>%
      layout(margin = list(l = 120, r = 20, t = 40, b = 20)) %>%
      config(displayModeBar = FALSE)
  })
  
  # -------------------------------------------------------------------------
  # Pir?mide poblacional
  # -------------------------------------------------------------------------
  output$piramide <- renderPlotly({
    df_pir <- df_filtrado() %>%
      filter(grupo_edad != "No informa", !is.na(sexo), sexo != "No informa") %>%
      mutate(grupo_edad = factor(grupo_edad,
                                 levels = c("15-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+")))
    
    df_pir <- df_pir %>%
      group_by(grupo_edad, sexo, estado_laboral) %>%
      summarise(n = n(), .groups = "drop") %>%
      group_by(grupo_edad, sexo) %>%
      mutate(prop = n / sum(n) * 100) %>%
      ungroup()
    
    df_pir_plot <- df_pir %>%
      mutate(prop_plot = ifelse(sexo == "Masculino", -prop, prop))
    
    p <- ggplot(df_pir_plot, aes(x = grupo_edad, y = prop_plot, fill = estado_laboral,
                                 text = paste0(estado_laboral, "<br>", abs(round(prop, 1)), "%"))) +
      geom_col(position = "stack", width = 0.7) +
      coord_flip() +
      scale_fill_manual(values = PAL_ESTADO, name = "Condición") +
      scale_y_continuous(labels = function(x) paste0(abs(x), "%")) +
      labs(x = "Grupo de edad", y = "Porcentaje") +
      tema_app +
      theme(legend.position = "top")
    
    ggplotly(p, tooltip = "text", height = 350) %>%
      layout(legend = list(orientation = "h", x = 0.5, y = 1.1, xanchor = "center"),
             margin = list(t = 60, b = 40)) %>%
      config(displayModeBar = FALSE)
  })
  
  # -------------------------------------------------------------------------
  # Serie de tiempo
  # -------------------------------------------------------------------------
  output$serie_tiempo <- renderPlotly({
    df_ts <- df_ft %>%
      filter(!is.na(sexo), sexo != "No informa",
             !is.na(grupo_edad), grupo_edad != "No informa") %>%
      group_by(ANIO, sexo, grupo_edad) %>%
      summarise(tasa = round(100 * mean(estado_laboral == "Desocupado"), 2), .groups = "drop") %>%
      mutate(grupo_edad = factor(grupo_edad,
                                 levels = c("15-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+")))
    
    top_grupos <- df_ts %>%
      group_by(grupo_edad) %>%
      summarise(prom = mean(tasa, na.rm = TRUE), .groups = "drop") %>%
      arrange(desc(prom)) %>%
      slice_head(n = 3) %>%
      pull(grupo_edad)
    
    df_ts_filt <- df_ts %>% filter(grupo_edad %in% top_grupos)
    
    p <- ggplot(df_ts_filt, aes(x = ANIO, y = tasa, color = sexo, linetype = grupo_edad,
                                group = interaction(sexo, grupo_edad),
                                text = paste0(sexo, "", grupo_edad, "<br>Año: ", ANIO, "<br>Tasa: ", tasa, "%"))) +
      geom_line(linewidth = 1.2) +
      geom_point(size = 2) +
      scale_color_manual(values = PAL_SEXO, name = "Sexo") +
      scale_y_continuous(labels = function(x) paste0(x, "%")) +
      labs(x = "Año", y = "Tasa de desempleo", linetype = "Grupo de edad") +
      tema_app +
      theme(legend.position = "top")
    
    ggplotly(p, tooltip = "text", height = 350) %>%
      layout(legend = list(orientation = "h", x = 0.5, y = 1.15, xanchor = "center"),
             margin = list(t = 70, b = 40)) %>%
      config(displayModeBar = FALSE)
  })
  
  # -------------------------------------------------------------------------
  # Predicci?n (reactiva)
  # -------------------------------------------------------------------------
  prediccion <- eventReactive(input$btn_predecir, {
    nuevo <- df_template[1, ]
    
    nuevo$sexo <- factor(input$in_sexo, levels = levels(df_template$sexo))
    nuevo$grupo_edad <- factor(input$in_grupo_edad, levels = levels(df_template$grupo_edad))
    nuevo$nivel_educativo <- factor(input$in_nivel_educativo, levels = levels(df_template$nivel_educativo))
    nuevo$etnia <- factor(input$in_etnia, levels = levels(df_template$etnia))
    nuevo$region <- factor(input$in_region, levels = levels(df_template$region))
    nuevo$zona <- factor(input$in_zona, levels = levels(df_template$zona))
    nuevo$tenencia <- factor(input$in_tenencia, levels = levels(df_template$tenencia))
    
    nuevo$discapacidad <- as.integer(input$in_discapacidad)
    nuevo$jefe_hogar <- as.integer(input$in_jefe_hogar)
    nuevo$mayor_18 <- ifelse(input$in_grupo_edad == "15-17", 0L, 1L)
    nuevo$estrato <- as.integer(input$in_estrato)
    nuevo$hacinamiento <- input$in_hacinamiento
    nuevo$servicios_basicos_score <- as.integer(input$in_servicios)
    nuevo$n_menores_15 <- as.integer(input$in_n_menores)
    nuevo$n_mayores_65 <- as.integer(input$in_n_mayores)
    nuevo$razon_dependencia <- (input$in_n_menores + input$in_n_mayores) / max(1, 2)
    nuevo$inclusion_fin_score <- as.integer(input$in_inclusion)
    nuevo$sin_producto_fin <- as.integer(input$in_sin_producto)
    nuevo$transferencias_gov <- as.integer(input$in_transferencias)
    nuevo$recibe_remesas <- as.integer(input$in_remesas)
    
    df_para_predecir <- bind_rows(df_template, nuevo)
    X_full <- model.matrix(y ~ . - 1, data = df_para_predecir)
    X_nuevo <- X_full[nrow(X_full), , drop = FALSE]
    
    cols_faltantes <- setdiff(cols_modelo, colnames(X_nuevo))
    for (c in cols_faltantes) X_nuevo <- cbind(X_nuevo, setNames(data.frame(0), c))
    
    X_nuevo <- X_nuevo[, cols_modelo, drop = FALSE]
    X_nuevo <- as.matrix(X_nuevo)
    
    list(prob = as.numeric(predict(modelo_xgb, X_nuevo)))
  })
  
  # -------------------------------------------------------------------------
  # Output: Caja de probabilidad (CORREGIDO)
  # -------------------------------------------------------------------------
  output$caja_probabilidad <- renderUI({
    # Si el bot?n no se ha presionado a?n, mostrar marcador de posici?n
    if (is.null(input$btn_predecir) || input$btn_predecir == 0) {
      return(
        div(
          div(class = "caja-probabilidad bajo",
              div(class = "num", " %"),
              div(class = "label", "Probabilidad estimada de desempleo")
          ),
          div(style = "margin-top: 15px;",
              div(style = "display: flex; justify-content: space-between; font-size: 11px; color: #888;",
                  span("0%"), span("50%"), span("100%")),
              div(class = "barra-tiempo",
                  div(class = "relleno", id = "barra-progreso", style = "width: 0%;")
              ),
              div(id = "etiqueta-riesgo", class = "etiqueta-riesgo",
                  style = "text-align: center; color: #888;",
                  "Presiona 'Calcular probabilidad'")
          )
        )
      )
    }
    
    # Si ya se presion?, calcular y mostrar resultado
    req(input$btn_predecir)
    p_pct <- prediccion()$prob * 100
    clase <- if (p_pct >= 50) "alto" else if (p_pct >= 25) "medio" else "bajo"
    nivel <- if (p_pct >= THRESHOLD_OPTIMO * 100) "ALTO RIESGO"
    else if (p_pct >= 25) "RIESGO MODERADO" else "BAJO RIESGO"
    
    # Color seg?n clase
    color_texto <- if (clase == "alto") "#D7191C"
    else if (clase == "medio") "#E89B40"
    else "#2E7D32"
    
    tagList(
      div(class = paste("caja-probabilidad", clase),
          div(class = "num", paste0(round(p_pct, 1), "%")),
          div(class = "label", "Probabilidad estimada de desempleo"),
          div(style = "font-size: 14px; color: #4A6072; font-weight: 600; margin-top: 4px;", nivel),
          div(style = "font-size: 11px; color: #6B7B85; margin-top: 6px;",
              paste0("Umbral óptimo: ", round(THRESHOLD_OPTIMO * 100, 1), "%"))
      ),
      div(style = "margin-top: 15px;",
          div(style = "display: flex; justify-content: space-between; font-size: 11px; color: #888;",
              span("0%"), span("50%"), span("100%")),
          div(class = "barra-tiempo",
              div(class = paste("relleno", clase), id = "barra-progreso",
                  style = sprintf("width: %s%%;", round(p_pct, 1)))
          ),
          div(id = "etiqueta-riesgo", class = "etiqueta-riesgo",
              style = sprintf("text-align: center; color: %s;", color_texto),
              nivel)
      )
    )
  })
  
  # -------------------------------------------------------------------------
  # Output: SHAP importancia
  # -------------------------------------------------------------------------
  output$shap_importancia <- renderUI({
    # Marcador de posici?n si no se ha calculado
    if (is.null(input$btn_predecir) || input$btn_predecir == 0) {
      return(
        div(style = "padding: 20px; text-align: center; color: #888;",
            "Completa el perfil y presiona 'Calcular probabilidad' para ver la importancia de variables.")
      )
    }
    
    req(input$btn_predecir)
    
    perfil_actual <- list(
      sexo = input$in_sexo,
      grupo_edad = input$in_grupo_edad,
      nivel_educativo = input$in_nivel_educativo,
      etnia = input$in_etnia,
      region = input$in_region,
      zona = input$in_zona,
      tenencia = input$in_tenencia,
      estrato = input$in_estrato,
      inclusion_fin_score = input$in_inclusion,
      servicios_basicos_score = input$in_servicios,
      discapacidad = as.integer(input$in_discapacidad)
    )
    
    nombres_legibles <- c(
      sexo = "Sexo",
      grupo_edad = "Edad",
      nivel_educativo = "Nivel educativo",
      etnia = "Etnia",
      region = "Departamento",
      zona = "Zona",
      tenencia = "Tenencia vivienda",
      estrato = "Estrato",
      inclusion_fin_score = "Inclusión financiera",
      servicios_basicos_score = "Servicios básicos",
      discapacidad = "Discapacidad"
    )
    
    impactos <- data.frame(
      variable = character(),
      valor = character(),
      diff = numeric(),
      stringsAsFactors = FALSE
    )
    
    for (var in names(perfil_actual)) {
      tabla <- tasas_marginales[[var]]
      if (is.null(tabla)) next
      tasa_v <- tabla[as.character(perfil_actual[[var]])]
      if (length(tasa_v) == 0 || is.na(tasa_v)) next
      impactos <- rbind(impactos, data.frame(
        variable = nombres_legibles[var],
        valor = as.character(perfil_actual[[var]]),
        diff = as.numeric(tasa_v) - TASA_NACIONAL,
        stringsAsFactors = FALSE
      ))
    }
    
    if (nrow(impactos) == 0) {
      return(p("No se pudo calcular la importancia de variables."))
    }
    
    impactos <- impactos[order(-abs(impactos$diff)), ]
    impactos <- head(impactos, 6)
    
    max_abs <- max(abs(impactos$diff), 0.1)
    
    tagList(
      lapply(1:nrow(impactos), function(i) {
        f <- impactos[i, ]
        pct <- abs(f$diff) / max_abs * 100
        clase <- if (f$diff > 0) "shap-positivo" else "shap-negativo"
        signo <- if (f$diff > 0) "+" else ""
        div(class = "shap-item",
            div(class = "shap-nombre", strong(f$variable), "", f$valor),
            div(class = "shap-barra", style = sprintf("width: %s%%; background: %s;",
                                                      pct, if (f$diff > 0) "#E89B97" else "#A8C5E0")),
            div(class = "shap-valor", style = if (f$diff > 0) "color: #D7191C;" else "color: #2E7D32;",
                paste0(signo, round(f$diff, 1), " pp"))
        )
      }),
      div(style = "font-size: 11px; color: #888; margin-top: 10px; text-align: center;",
          "Rojo = aumenta riesgo  Azul = reduce riesgo")
    )
  })
}

# =============================================================================
# LANZAMIENTO
# =============================================================================
shinyApp(ui, server)