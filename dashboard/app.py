# dashboard/app.py
import shiny as sh
from shiny import ui, reactive, render
import pandas as pd
import plotly.express as px
import httpx
import asyncio

# --- CONFIGURACIÓN DE LA API ---
API_URL = "http://13.220.102.90:8001/predict"  # <-- CAMBIA ESTO POR TU IP REAL DE EC2

# --- INTERFAZ DE USUARIO ---
app_ui = ui.page_navbar(
    ui.nav_panel(
        "📈 Panorama General",
        ui.layout_sidebar(
            ui.panel_sidebar(
                ui.h4("Filtros"),
                ui.input_select("year_filter", "Año", choices=["2023", "2024", "2025"]),
                ui.input_select("depto_filter", "Departamento", choices=["Antioquia", "Atlántico", "Bogotá D.C.", "Valle", "Amazonas"])
            ),
            ui.panel_main(
                # Usamos output_ui para que Plotly se renderice correctamente (no render.plot)
                ui.row(
                    ui.column(6, ui.output_ui("map_card")),
                    ui.column(6, ui.output_ui("line_card"))
                )
            )
        )
    ),
    ui.nav_panel(
        "🔍 Predicción Individual",
        ui.layout_sidebar(
            ui.panel_sidebar(
                ui.h4("Perfil del Solicitante"),
                # Sección 1: Datos personales
                ui.input_select("sexo", "Sexo", choices=["Mujer", "Hombre"]),
                ui.input_select("grupo_edad", "Grupo de Edad", choices=["15-24", "25-34", "35-44", "45-54", "55+"]),
                ui.input_select("nivel_educativo", "Nivel Educativo", choices=["Ninguno", "Primaria", "Secundaria", "Universitario", "Posgrado"]),
                ui.input_select("etnia", "Etnia", choices=["Ninguna", "Indígena", "Afrocolombiano", "Raizal", "Palenquero"]),
                ui.input_select("discapacidad", "Discapacidad", choices=["No", "Sí"]),
                ui.input_select("jefe_hogar", "Jefe del Hogar", choices=["No", "Sí"]),
                
                # Sección 2: Ubicación y vivienda
                ui.input_select("region", "Región", choices=["Amazonía", "Andina", "Caribe", "Orinoquía", "Pacífico"]),
                ui.input_select("zona", "Zona", choices=["Urbana", "Rural"]),
                ui.input_select("estrato", "Estrato", choices=["1", "2", "3", "4", "5", "6"]),
                ui.input_select("hacinamiento", "Hacinamiento", choices=["No", "Sí"]),
                ui.input_select("tenencia", "Tenencia de Vivienda", choices=["Propia", "Arrendada", "Familiar"]),
                ui.input_slider("servicios_basicos_score", "Score Servicios Básicos (0-100)", min=0, max=100, value=80),

                # Sección 3: Finanzas y hogar
                ui.input_slider("inclusion_fin_score", "Score Inclusión Financiera (0-100)", min=0, max=100, value=50),
                ui.input_select("sin_producto_fin", "Sin Producto Financiero", choices=["No", "Sí"]),
                ui.input_select("transferencias_gov", "Transferencias del Gobierno", choices=["No", "Sí"]),
                ui.input_select("recibe_remesas", "Recibe Remesas", choices=["No", "Sí"]),
                ui.input_numeric("n_menores_15", "N° Menores de 15 años", value=0),
                ui.input_numeric("n_mayores_65", "N° Mayores de 65 años", value=0),
                ui.input_numeric("razon_dependencia", "Razón de Dependencia", value=1.0),
                
                ui.input_action_button("btn_predict", "🔮 Calcular Probabilidad", class_="btn-primary")
            ),
            ui.panel_main(
                ui.output_ui("prediction_result")
            )
        )
    ),
    title="📊 Panel Riesgo de Desempleo - GEIH",
    id="navbar"
)

# --- LÓGICA DEL SERVIDOR ---
def server(input, output, session):
    
    # 1. Lógica de la Pestaña Panorama (Visualizaciones con Plotly directamente en render.ui)
    @output
    @render.ui
    def map_card():
        df = pd.DataFrame({
            'Departamento': ['Amazonas', 'Antioquia', 'Atlántico', 'Bogotá D.C.', 'Valle del Cauca'],
            'Tasa_Desempleo': [8.5, 10.2, 12.1, 9.8, 11.4]
        })
        fig = px.choropleth(df, locations='Departamento', locationmode='USA-states', 
                            color='Tasa_Desempleo', scope='south america', 
                            title='Tasa de Desempleo Departamental (%)')
        return ui.card(ui.card_header("Mapa de Calor"), fig)

    @output
    @render.ui
    def line_card():
        df_series = pd.DataFrame({
            'Fecha': pd.date_range(start='2023-01-01', periods=12, freq='M'),
            'Tasa': [11.5, 10.8, 9.9, 10.2, 11.0, 11.8, 12.1, 11.5, 10.9, 10.5, 10.8, 11.2]
        })
        fig = px.line(df_series, x='Fecha', y='Tasa', title='Evolución Mensual del Desempleo')
        return ui.card(ui.card_header("Serie de Tiempo"), fig)

    # 2. Lógica de la Pestaña Predicción (Consumo de API)
    @reactive.event(input.btn_predict)
    async def get_prediction():
        # Construir el payload según tu DataInputSchema
        payload = [{
            "sexo": input.sexo(),
            "grupo_edad": input.grupo_edad(),
            "nivel_educativo": input.nivel_educativo(),
            "etnia": input.etnia(),
            "discapacidad": 1 if input.discapacidad() == "Sí" else 0,
            "jefe_hogar": 1 if input.jefe_hogar() == "Sí" else 0,
            "mayor_18": 1, 
            "region": input.region(),
            "zona": input.zona(),
            "estrato": input.estrato(),
            "hacinamiento": 1 if input.hacinamiento() == "Sí" else 0,
            "servicios_basicos_score": input.servicios_basicos_score(),
            "tenencia": input.tenencia(),
            "n_menores_15": input.n_menores_15(),
            "n_mayores_65": input.n_mayores_65(),
            "razon_dependencia": input.razon_dependencia(),
            "inclusion_fin_score": input.inclusion_fin_score(),
            "sin_producto_fin": 1 if input.sin_producto_fin() == "Sí" else 0,
            "transferencias_gov": 1 if input.transferencias_gov() == "Sí" else 0,
            "recibe_remesas": 1 if input.recibe_remesas() == "Sí" else 0
        }]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(API_URL, json=payload, timeout=10.0)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Error en API: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    @output
    @render.ui
    def prediction_result():
        result = get_prediction()
        
        if "error" in result:
            return ui.card(
                ui.card_header("❌ Error"),
                f"No se pudo conectar con la API. Verifica que el servidor esté corriendo y el puerto 8001 abierto. Detalle: {result['error']}"
            )
        
        # Asumiendo que tu API devuelve una lista con [{"probability": X, "class": Y}]
        data = result[0]
        prob = data['probability']
        clase_text = "Desempleado" if data['class'] == 1 else "Ocupado/Inactivo"
        
        return ui.card(
            ui.card_header("📊 Resultado de la Predicción"),
            ui.h3(f"Clase Predicha: {clase_text}", style=f"color: {'red' if data['class'] == 1 else 'green'}"),
            ui.h4(f"Probabilidad de Desempleo: {prob*100:.2f}%"),
            ui.div(
                ui.tags.progress(
                    value=prob*100, max=100, 
                    style="width: 100%; height: 30px; background-color: #e9ecef; border-radius: 5px;"
                ),
                style="margin-top: 15px;"
            )
        )

# --- EJECUCIÓN ---
app = sh.App(app_ui, server)