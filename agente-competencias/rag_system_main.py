import streamlit as st
import fitz  # PyMuPDF
import json
import re
import pandas as pd
import os
import tempfile
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

# Importaciones para el sistema RAG
from sentence_transformers import SentenceTransformer, util
import faiss
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pdfplumber  # Mejor herramienta para extraer tablas que tabula
from io import BytesIO

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURACIÓN Y CARGA DE MODELOS
# ==============================================================================

@st.cache_resource
def cargar_modelo_embeddings():
    """Carga el modelo de embeddings para búsqueda semántica"""
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def cargar_modelo_llm():
    """Carga el modelo LLM liviano para análisis de competencias"""
    try:
        # Usar Flan-T5 small como modelo liviano
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Crear pipeline de text-generation
        llm_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.3,
            do_sample=True
        )
        return llm_pipeline
    except Exception as e:
        logger.error(f"Error cargando LLM: {e}")
        return None


@st.cache_data
def cargar_jsons():
    """Carga los archivos JSON de competencias"""
    try:
        with open("competenciascursos.json", encoding="utf-8") as f:
            competencias_cursos = json.load(f)
        with open("competencias.json", encoding="utf-8") as f:
            competencias = json.load(f)
        with open("abet_es.json", encoding="utf-8") as f:
            abet_es = json.load(f)
        return competencias_cursos, competencias, abet_es
    except Exception as e:
        logger.error(f"Error cargando JSONs: {e}")
        return {}, {}, {}


# ==============================================================================
# SISTEMA DE VECTOR DATABASE CON FAISS
# ==============================================================================

class VectorDatabase:
    def __init__(self, model):
        self.model = model
        self.index = None
        self.documents = []
        self.metadata = []

    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Añade documentos al índice vectorial"""
        embeddings = self.model.encode(documents)

        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product para cosine similarity

        # Normalizar embeddings para cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))

        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Busca documentos similares"""
        if self.index is None:
            return []

        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding.astype('float32'), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx],
                    float(score),
                    self.metadata[idx]
                ))
        return results

    def save(self, filepath: str):
        """Guarda el índice y metadatos"""
        faiss.write_index(self.index, f"{filepath}.faiss")
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)

    def load(self, filepath: str):
        """Carga el índice y metadatos"""
        self.index = faiss.read_index(f"{filepath}.faiss")
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']


# ==============================================================================
# EXTRACCIÓN MEJORADA DE TEXTO Y TABLAS
# ==============================================================================

def extraer_texto_pdf(uploaded_file) -> str:
    """Extrae texto del PDF usando PyMuPDF"""
    texto = ""
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                texto += page.get_text()
        return texto
    except Exception as e:
        logger.error(f"Error extrayendo texto: {e}")
        return ""


def extraer_tablas_pdf_mejorado(uploaded_file) -> List[pd.DataFrame]:
    """
    Extrae tablas usando pdfplumber - más confiable que tabula
    """
    tablas = []
    try:
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Usar pdfplumber para extraer tablas
        with pdfplumber.open(tmp_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extraer tablas de la página
                page_tables = page.extract_tables()

                for table_num, table in enumerate(page_tables):
                    if table and len(table) > 1:  # Verificar que tenga datos
                        try:
                            # Convertir a DataFrame con limpieza robusta
                            df = crear_dataframe_limpio(table)

                            # Verificar si parece una tabla de competencias/resultados
                            if es_tabla_competencias(df):
                                tablas.append(df)
                                logger.info(f"Tabla extraída de página {page_num + 1}")

                        except Exception as e:
                            logger.warning(f"Error procesando tabla en página {page_num + 1}: {e}")

        # Limpiar archivo temporal
        os.unlink(tmp_path)
        return tablas

    except Exception as e:
        logger.error(f"Error extrayendo tablas: {e}")
        return []


def crear_dataframe_limpio(table_data) -> pd.DataFrame:
    """
    Crea un DataFrame limpio compatible con Streamlit/PyArrow
    """
    try:
        # Separar headers de datos
        if len(table_data) < 2:
            return pd.DataFrame()

        headers = table_data[0]
        data_rows = table_data[1:]

        # Limpiar headers
        headers_limpios = []
        for i, header in enumerate(headers):
            if header is None or str(header).strip() == '':
                headers_limpios.append(f"Columna_{i + 1}")
            else:
                # Limpiar caracteres problemáticos
                header_limpio = str(header).strip()
                header_limpio = re.sub(r'[^\w\s\-_()]', '', header_limpio)
                headers_limpios.append(header_limpio if header_limpio else f"Columna_{i + 1}")

        # Limpiar datos
        data_limpia = []
        for row in data_rows:
            row_limpia = []
            for cell in row:
                if cell is None:
                    row_limpia.append("")
                else:
                    # Convertir a string y limpiar
                    cell_str = str(cell).strip()
                    # Reemplazar caracteres problemáticos
                    cell_str = cell_str.replace('\n', ' ').replace('\r', ' ')
                    # Limitar longitud para evitar problemas
                    if len(cell_str) > 500:
                        cell_str = cell_str[:500] + "..."
                    row_limpia.append(cell_str)
            data_limpia.append(row_limpia)

        # Crear DataFrame
        df = pd.DataFrame(data_limpia, columns=headers_limpios)

        # Asegurar que todas las columnas sean strings para compatibilidad
        for col in df.columns:
            df[col] = df[col].astype(str)

        # Eliminar filas completamente vacías
        df = df.replace('', None)
        df = df.dropna(how='all')
        df = df.fillna('')

        # Validar que el DataFrame no esté vacío
        if df.empty or len(df.columns) == 0:
            return pd.DataFrame()

        return df

    except Exception as e:
        logger.error(f"Error creando DataFrame limpio: {e}")
        return pd.DataFrame()


def es_tabla_competencias(df: pd.DataFrame) -> bool:
    """Verifica si una tabla parece contener datos de competencias"""
    if df.empty or len(df.columns) < 2:
        return False

    # Buscar palabras clave en las columnas y primeras filas
    keywords = ['competencia', 'desempeño', 'porcentaje', 'raec', 'estudiantes', 'resultado', 'aprobado', 'nota']

    # Texto de headers
    headers_text = ' '.join(df.columns).lower()

    # Texto de primeras filas
    sample_text = ""
    if len(df) > 0:
        sample_text = ' '.join(df.iloc[0:min(3, len(df))].astype(str).values.flatten()).lower()

    combined_text = headers_text + ' ' + sample_text

    return any(keyword in combined_text for keyword in keywords)


# ==============================================================================
# ANÁLISIS CON LLM
# ==============================================================================

class AnalizadorCompetencias:
    def __init__(self, llm_pipeline, vector_db):
        self.llm = llm_pipeline
        self.vector_db = vector_db

    def identificar_competencias_tabla(self, tabla: pd.DataFrame) -> List[str]:
        """Identifica competencias en una tabla usando LLM"""
        if self.llm is None:
            return []

        # Convertir tabla a texto para el LLM
        tabla_texto = self.tabla_a_texto(tabla)

        prompt = f"""
        Analiza la siguiente tabla de resultados académicos y identifica qué competencias específicas se están evaluando.
        Busca códigos como: CE1, CE2, CG1, CG2, SP1, SP2, etc.

        Tabla:
        {tabla_texto}

        Responde solo con los códigos de competencias separados por comas:
        """

        try:
            respuesta = self.llm(prompt, max_length=100)[0]['generated_text']
            # Extraer códigos usando regex
            codigos = re.findall(r'\b[A-Z]{1,3}\d+\b', respuesta)
            return list(set(codigos))
        except Exception as e:
            logger.error(f"Error en análisis LLM: {e}")
            return []

    def comparar_competencias_con_llm(self, esperadas: List[str], detectadas: List[str]) -> Dict:
        """Compara competencias usando LLM para análisis más sofisticado"""
        if self.llm is None:
            return self.comparacion_simple(esperadas, detectadas)

        prompt = f"""
        Compara estas dos listas de competencias académicas:

        Competencias esperadas en PDA: {', '.join(esperadas)}
        Competencias detectadas en tabla: {', '.join(detectadas)}

        Analiza:
        1. ¿Cuáles coinciden exactamente?
        2. ¿Cuáles faltan en la tabla?
        3. ¿Hay competencias extra no esperadas?

        Responde en formato JSON:
        """

        try:
            respuesta = self.llm(prompt, max_length=200)[0]['generated_text']
            # Procesar respuesta del LLM
            return self.procesar_respuesta_comparacion(respuesta, esperadas, detectadas)
        except Exception as e:
            logger.error(f"Error en comparación LLM: {e}")
            return self.comparacion_simple(esperadas, detectadas)

    def tabla_a_texto(self, tabla: pd.DataFrame) -> str:
        """Convierte tabla a texto legible para el LLM"""
        if tabla.empty:
            return ""

        # Tomar solo las primeras filas y columnas relevantes
        muestra = tabla.head(10)
        return muestra.to_string(index=False)

    def comparacion_simple(self, esperadas: List[str], detectadas: List[str]) -> Dict:
        """Comparación básica sin LLM"""
        esperadas_set = set(esperadas)
        detectadas_set = set(detectadas)

        return {
            'coincidencias': list(esperadas_set.intersection(detectadas_set)),
            'faltantes': list(esperadas_set - detectadas_set),
            'extras': list(detectadas_set - esperadas_set),
            'porcentaje_coincidencia': len(esperadas_set.intersection(detectadas_set)) / len(
                esperadas_set) * 100 if esperadas else 0
        }

    def procesar_respuesta_comparacion(self, respuesta: str, esperadas: List[str], detectadas: List[str]) -> Dict:
        """Procesa la respuesta del LLM y extrae información estructurada"""
        # Si falla el procesamiento del LLM, usar comparación simple
        try:
            # Intentar extraer códigos de la respuesta
            codigos_encontrados = re.findall(r'\b[A-Z]{1,3}\d+\b', respuesta)

            esperadas_set = set(esperadas)
            detectadas_set = set(detectadas)

            return {
                'coincidencias': list(esperadas_set.intersection(detectadas_set)),
                'faltantes': list(esperadas_set - detectadas_set),
                'extras': list(detectadas_set - esperadas_set),
                'porcentaje_coincidencia': len(esperadas_set.intersection(detectadas_set)) / len(
                    esperadas_set) * 100 if esperadas else 0,
                'analisis_llm': respuesta[:200]  # Resumen del análisis
            }
        except:
            return self.comparacion_simple(esperadas, detectadas)


# ==============================================================================
# FUNCIONES DE DETECCIÓN (EXISTENTES MEJORADAS)
# ==============================================================================

def detectar_codigo_curso(texto: str, codigos_disponibles: List[str]) -> Optional[str]:
    """Detecta código de curso en el texto"""
    texto_lower = texto.lower()
    for codigo in codigos_disponibles:
        if codigo.lower() in texto_lower:
            return codigo
    return None


def ampliar_descripcion_saberpro(cod: str, descripcion: str) -> str:
    """Amplía descripciones para SABER PRO"""
    descripciones_ampliadas = {
        "SP1": "Capacidad para realizar razonamiento cuantitativo, resolver problemas numéricos y analizar datos utilizando métodos matemáticos y estadísticos.",
        "SP2": "Habilidad para comprender, interpretar y analizar textos complejos, identificando ideas principales y secundarias en contextos variados.",
        "SP3": "Competencia en la redacción de textos claros, coherentes y estructurados, utilizando un lenguaje formal y adaptado al contexto.",
        "SP4": "Capacidad para actuar como ciudadano responsable, tomar decisiones informadas y comprender las dinámicas sociales y éticas en contextos democráticos.",
        "SP5": "Demostrar competencia en el idioma inglés, incluyendo habilidades de comprensión lectora, escritura y comunicación oral en una segunda lengua."
    }
    return descripciones_ampliadas.get(cod, descripcion)


def describir_competencias(codigos: List[str], tipo: str, competencias: Dict, abet_es: Dict) -> List[
    Tuple[str, str, str]]:
    """Describe competencias basado en su tipo"""
    data = []
    if tipo == "abet":
        for cod in codigos:
            desc = None
            for abet_id, contenido in abet_es.items():
                indicadores = contenido.get("indicadores", {})
                if cod in indicadores:
                    desc = indicadores[cod]
                    break
            data.append((tipo.upper(), cod, desc if desc else "No encontrada"))
    else:
        tipo_map = {
            "especificas": "competencias especificas",
            "genericas": "competencias genericas",
            "saberpro": "SABER PRO",
            "dimension": "dimensiones"
        }
        for cod in codigos:
            desc = competencias[tipo_map[tipo]].get(cod, "No encontrada")
            if tipo == "saberpro":
                desc = ampliar_descripcion_saberpro(cod, desc)
            data.append((tipo.capitalize(), cod, desc))
    return data


def detectar_codigos_en_texto(texto: str, competencias: Dict, abet_es: Dict) -> List[str]:
    """Detecta códigos de competencias en texto"""
    encontrados = set()
    texto = texto.lower()

    codigos = []
    for tipo in ["competencias especificas", "competencias genericas", "SABER PRO", "dimensiones"]:
        codigos.extend(competencias[tipo].keys())

    for ab in abet_es.values():
        codigos.extend(ab.get("indicadores", {}).keys())

    for cod in codigos:
        variantes = [cod.lower(), f"o{cod.lower()}", cod.replace(".", "")]
        if cod == "SP5":
            variantes.extend(["inglés", "idioma inglés", "segunda lengua", "ingles"])
        for var in variantes:
            if re.search(rf"\b{re.escape(var)}\b", texto):
                encontrados.add(cod)
                break
    return list(encontrados)


def detectar_por_similitud(texto: str, competencias_esperadas: Dict, model, threshold: float = 0.70) -> List[str]:
    """Detección semántica con embeddings"""
    frases_pda = [line.strip() for line in texto.split("\n") if len(line.strip()) > 10]
    if not frases_pda:
        return []

    embeddings_pda = model.encode(frases_pda, convert_to_tensor=True)
    resultados = set()

    for cod, descripcion in competencias_esperadas.items():
        current_threshold = 0.60 if cod.startswith("SP") else threshold
        emb_target = model.encode(descripcion, convert_to_tensor=True)
        scores = util.cos_sim(emb_target, embeddings_pda)[0]
        max_score = max(scores).item()
        if max_score > current_threshold:
            resultados.add(cod)
            logger.info(f"Código: {cod}, Máximo score: {max_score}")

    return list(resultados)


# ==============================================================================
# MÓDULOS PRINCIPALES DE LA APLICACIÓN
# ==============================================================================

def main_pda():
    """Módulo para análisis de PDA mejorado con RAG"""
    st.markdown("### 📋 Análisis de PDA con Sistema RAG")
    st.markdown("Sube un **PDA en PDF** y el sistema detectará las competencias usando IA avanzada.")

    uploaded_file = st.file_uploader("📤 Subir archivo PDF del PDA", type="pdf", key="pda_uploader")

    if uploaded_file:
        # Cargar modelos y datos
        modelo_embeddings = cargar_modelo_embeddings()
        llm_pipeline = cargar_modelo_llm()
        competencias_cursos, competencias, abet_es = cargar_jsons()

        # Inicializar sistema RAG
        vector_db = VectorDatabase(modelo_embeddings)
        analizador = AnalizadorCompetencias(llm_pipeline, vector_db)

        with st.spinner("Procesando documento..."):
            # Extraer texto
            texto = extraer_texto_pdf(uploaded_file).lower()

            # Detectar código de curso
            codigo_detectado = detectar_codigo_curso(texto, competencias_cursos.keys())

            if not codigo_detectado:
                st.error("❌ No se detectó un código de curso válido.")
                return

            st.success(f"✅ Código detectado: `{codigo_detectado}`")

            # Obtener competencias esperadas
            comp_curso = competencias_cursos.get(codigo_detectado, {})
            esperadas = {}

            for tipo in ["especificas", "genericas", "saberpro", "abet", "dimension"]:
                codigos = comp_curso.get(tipo, [])
                for _, cod, desc in describir_competencias(codigos, tipo, competencias, abet_es):
                    if cod and desc:
                        esperadas[cod] = desc

            # Indexar competencias en vector database
            if esperadas:
                docs = [f"{cod}: {desc}" for cod, desc in esperadas.items()]
                metadata = [{"codigo": cod, "tipo": "esperada"} for cod in esperadas.keys()]
                vector_db.add_documents(docs, metadata)

            # Detección múltiple de competencias
            detectadas_codigos = detectar_codigos_en_texto(texto, competencias, abet_es)
            detectadas_semantica = detectar_por_similitud(texto, esperadas, modelo_embeddings)
            detectadas_totales = set(detectadas_codigos).union(set(detectadas_semantica))

        # Mostrar resultados
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📌 Competencias Esperadas")
            if esperadas:
                df_esperadas = pd.DataFrame([(k, v[:100] + "..." if len(v) > 100 else v)
                                             for k, v in esperadas.items()],
                                            columns=["Código", "Descripción"])
                st.dataframe(df_esperadas, use_container_width=True)
            else:
                st.warning("No se encontraron competencias esperadas para este curso.")

        with col2:
            st.subheader("🔍 Competencias Detectadas")
            if detectadas_totales:
                st.write(f"**Total detectadas:** {len(detectadas_totales)}")
                for cod in sorted(detectadas_totales):
                    metodo = "📝 Código" if cod in detectadas_codigos else "🧠 Semántica"
                    st.write(f"• {cod} ({metodo})")
            else:
                st.warning("No se detectaron competencias.")

        # Análisis comparativo
        if esperadas and detectadas_totales:
            st.subheader("📊 Análisis Comparativo RAG")

            with st.spinner("Analizando con IA..."):
                comparacion = analizador.comparar_competencias_con_llm(
                    list(esperadas.keys()),
                    list(detectadas_totales)
                )

            # Métricas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Coincidencias", len(comparacion['coincidencias']))
            with col2:
                st.metric("Faltantes", len(comparacion['faltantes']))
            with col3:
                st.metric("Extras", len(comparacion['extras']))
            with col4:
                st.metric("% Coincidencia", f"{comparacion['porcentaje_coincidencia']:.1f}%")

            # Detalles
            if comparacion['coincidencias']:
                st.success(f"✅ **Coincidencias:** {', '.join(comparacion['coincidencias'])}")
            if comparacion['faltantes']:
                st.error(f"❌ **Faltantes:** {', '.join(comparacion['faltantes'])}")
            if comparacion['extras']:
                st.warning(f"⚠️ **Extras:** {', '.join(comparacion['extras'])}")

            # Análisis del LLM si está disponible
            if 'analisis_llm' in comparacion:
                st.info(f"🤖 **Análisis IA:** {comparacion['analisis_llm']}")


def main_reflexion_rae():
    """Módulo para análisis de Reflexión RAE mejorado con mejor manejo de errores"""
    st.markdown("### 📊 Análisis de Reflexión RAE con IA")
    st.markdown("Sube un **PDF de Reflexión RAE** para extraer y analizar tablas de resultados.")

    uploaded_file = st.file_uploader("📤 Subir archivo PDF de Reflexión RAE", type="pdf", key="rae_uploader")

    if uploaded_file:
        # Cargar modelos
        modelo_embeddings = cargar_modelo_embeddings()
        llm_pipeline = cargar_modelo_llm()
        competencias_cursos, competencias, abet_es = cargar_jsons()

        analizador = AnalizadorCompetencias(llm_pipeline, vector_db=None)

        with st.spinner("Procesando documento RAE..."):
            # Extraer texto para código de curso
            uploaded_file.seek(0)  # Reset file pointer
            texto = extraer_texto_pdf(uploaded_file)
            codigo_detectado = detectar_codigo_curso(texto, competencias_cursos.keys())

            if not codigo_detectado:
                st.error("❌ No se detectó un código de curso válido.")
                return

            st.success(f"✅ Código detectado: `{codigo_detectado}`")

            # Extraer tablas mejorado
            uploaded_file.seek(0)  # Reset file pointer
            tablas = extraer_tablas_pdf_mejorado(uploaded_file)

            if not tablas:
                st.error("❌ No se detectaron tablas válidas en el PDF.")
                st.info("💡 **Sugerencias:**")
                st.info("- Asegúrate de que el PDF contenga tablas con estructura clara")
                st.info("- Verifica que las tablas no sean imágenes")
                st.info("- Prueba con un PDF diferente")
                return

            st.success(f"✅ Se extrajeron {len(tablas)} tabla(s)")

        # Mostrar tablas extraídas con manejo de errores mejorado
        st.subheader("📊 Tablas Extraídas")

        competencias_detectadas_tablas = []

        for i, tabla in enumerate(tablas, 1):
            with st.expander(f"📋 Tabla {i} - {tabla.shape[0]} filas x {tabla.shape[1]} columnas"):
                try:
                    # Mostrar información de la tabla
                    st.write(f"**Columnas:** {', '.join(tabla.columns)}")

                    # Intentar mostrar el DataFrame
                    st.dataframe(tabla, use_container_width=True)

                    # Análisis IA de la tabla
                    if llm_pipeline:
                        with st.spinner(f"Analizando tabla {i} con IA..."):
                            competencias_tabla = analizador.identificar_competencias_tabla(tabla)

                        if competencias_tabla:
                            st.write(f"🤖 **Competencias identificadas por IA:** {', '.join(competencias_tabla)}")
                            competencias_detectadas_tablas.extend(competencias_tabla)
                        else:
                            st.write("🤖 **IA:** No se identificaron competencias específicas en esta tabla")

                except Exception as e:
                    st.error(f"❌ Error mostrando tabla {i}: {e}")
                    st.write("**Vista alternativa (primeras 5 filas):**")
                    try:
                        st.text(tabla.head().to_string())
                    except:
                        st.text("No se puede mostrar la tabla")

        # Comparación con competencias del curso
        if competencias_detectadas_tablas:
            st.subheader("🔄 Comparación RAE vs PDA")

            # Obtener competencias esperadas del curso
            comp_curso = competencias_cursos.get(codigo_detectado, {})
            esperadas_curso = []

            for tipo in ["especificas", "genericas", "saberpro", "abet", "dimension"]:
                esperadas_curso.extend(comp_curso.get(tipo, []))

            if esperadas_curso:
                comparacion = analizador.comparar_competencias_con_llm(
                    esperadas_curso,
                    competencias_detectadas_tablas
                )

                # Mostrar métricas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Competencias en RAE", len(set(competencias_detectadas_tablas)))
                with col2:
                    st.metric("Competencias en PDA", len(esperadas_curso))
                with col3:
                    st.metric("% Coincidencia", f"{comparacion['porcentaje_coincidencia']:.1f}%")

                # Alertas
                if comparacion['porcentaje_coincidencia'] > 80:
                    st.success("🎉 ¡Excelente alineación entre RAE y PDA!")
                elif comparacion['porcentaje_coincidencia'] > 60:
                    st.warning("⚠️ Alineación parcial - revisar competencias faltantes")
                else:
                    st.error("❌ Baja alineación - se requiere revisión profunda")

                # Detalles de la comparación
                if comparacion['faltantes']:
                    st.error(f"❌ **Competencias en PDA pero no en RAE:** {', '.join(comparacion['faltantes'])}")
                if comparacion['extras']:
                    st.info(f"ℹ️ **Competencias en RAE pero no en PDA:** {', '.join(comparacion['extras'])}")
            else:
                st.warning("No se encontraron competencias definidas para este curso.")

def main_vector_search():
    """Módulo para búsqueda vectorial"""
    st.markdown("### 🔍 Búsqueda Vectorial de Competencias")
    st.markdown("Busca competencias similares usando IA y embeddings semánticos.")

    # Cargar modelo
    modelo_embeddings = cargar_modelo_embeddings()
    competencias_cursos, competencias, abet_es = cargar_jsons()

    # Crear base de conocimiento
    if 'vector_db' not in st.session_state:
        with st.spinner("Inicializando base de conocimiento..."):
            st.session_state.vector_db = VectorDatabase(modelo_embeddings)

            # Indexar todas las competencias
            documentos = []
            metadatos = []

            for tipo, comp_dict in competencias.items():
                for cod, desc in comp_dict.items():
                    documentos.append(f"{cod}: {desc}")
                    metadatos.append({"codigo": cod, "tipo": tipo})

            st.session_state.vector_db.add_documents(documentos, metadatos)

            # Indexar ABET
            for abet_id, contenido in abet_es.items():
                indicadores = contenido.get("indicadores", {})
                for cod, desc in indicadores.items():
                    documentos.append(f"{cod}: {desc}")
                    metadatos.append({"codigo": cod, "tipo": "abet"})

            st.session_state.vector_db.add_documents(documentos[-len(indicadores):], metadatos[-len(indicadores):])

    # Interfaz de búsqueda
    query = st.text_input("🔍 Buscar competencias (describe lo que necesitas):",
                          placeholder="Ej: habilidades de programación en Python")

    num_results = st.slider("Número de resultados:", 1, 10, 5)

    if query:
        with st.spinner("Buscando..."):
            resultados = st.session_state.vector_db.search(query, k=num_results)

        if resultados:
            st.subheader(f"📋 Resultados para: '{query}'")

            for i, (documento, score, metadata) in enumerate(resultados, 1):
                with st.expander(f"#{i} - {metadata.get('codigo', 'N/A')} (Similitud: {score:.3f})"):
                    st.write(f"**Tipo:** {metadata.get('tipo', 'N/A')}")
                    st.write(f"**Contenido:** {documento}")
                    st.write(f"**Puntuación:** {score:.3f}")
        else:
            st.warning("No se encontraron resultados relevantes.")


def main_configuracion():
    """Módulo de configuración del sistema"""
    st.markdown("### ⚙️ Configuración del Sistema RAG")

    # Estado del sistema
    st.subheader("📊 Estado del Sistema")

    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            cargar_modelo_embeddings()
            st.metric("Modelo Embeddings", "✅ Cargado")
        except:
            st.metric("Modelo Embeddings", "❌ Error")

    with col2:
        llm_status = "✅ Cargado" if cargar_modelo_llm() is not None else "❌ Error"
        st.metric("Modelo LLM", llm_status)

    with col3:
        db_status = "✅ Activa" if 'vector_db' in st.session_state else "❌ No inicializada"
        st.metric("Vector DB", db_status)

    # Configuraciones
    st.subheader("🎛️ Parámetros")

    with st.expander("🔧 Configuración de Detección"):
        threshold_semantic = st.slider("Umbral semántico:", 0.0, 1.0, 0.70, 0.05)
        threshold_saberpro = st.slider("Umbral SABER PRO:", 0.0, 1.0, 0.60, 0.05)

        st.session_state.config = {
            'threshold_semantic': threshold_semantic,
            'threshold_saberpro': threshold_saberpro
        }

    with st.expander("💾 Gestión de Datos"):
        st.write("**Cargar/Guardar Vector Database**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Guardar Vector DB"):
                if 'vector_db' in st.session_state:
                    try:
                        st.session_state.vector_db.save("vector_db_competencias")
                        st.success("✅ Vector DB guardada exitosamente")
                    except Exception as e:
                        st.error(f"❌ Error guardando: {e}")
                else:
                    st.warning("No hay Vector DB para guardar")

        with col2:
            if st.button("📂 Cargar Vector DB"):
                try:
                    modelo = cargar_modelo_embeddings()
                    vector_db = VectorDatabase(modelo)
                    vector_db.load("vector_db_competencias")
                    st.session_state.vector_db = vector_db
                    st.success("✅ Vector DB cargada exitosamente")
                except Exception as e:
                    st.error(f"❌ Error cargando: {e}")

    # Estadísticas
    if 'vector_db' in st.session_state:
        st.subheader("📈 Estadísticas")

        vector_db = st.session_state.vector_db

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documentos indexados", len(vector_db.documents))
        with col2:
            tipos = [meta.get('tipo', 'N/A') for meta in vector_db.metadata]
            st.metric("Tipos únicos", len(set(tipos)))
        with col3:
            if vector_db.index:
                st.metric("Dimensión vectores", vector_db.index.d)


# ==============================================================================
# APLICACIÓN PRINCIPAL
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Sistema RAG - Análisis de Competencias",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎓 Sistema RAG para Análisis de Competencias Académicas")
    st.markdown("---")

    # Sidebar para navegación
    with st.sidebar:
        st.title("📋 Navegación")

        pagina = st.radio(
            "Selecciona módulo:",
            [
                "🏠 Inicio",
                "📋 Análisis PDA",
                "📊 Análisis RAE",
                "🔍 Búsqueda Vectorial",
                "⚙️ Configuración"
            ]
        )

        st.markdown("---")
        st.markdown("### 🔄 Estado del Sistema")

        # Verificar estado de modelos
        try:
            modelo_status = "🟢 OK"
            cargar_modelo_embeddings()
        except:
            modelo_status = "🔴 Error"

        try:
            llm_status = "🟢 OK" if cargar_modelo_llm() else "🔴 Error"
        except:
            llm_status = "🔴 Error"

        st.write(f"**Embeddings:** {modelo_status}")
        st.write(f"**LLM:** {llm_status}")
        st.write(f"**Vector DB:** {'🟢 OK' if 'vector_db' in st.session_state else '🟡 No inicializada'}")

        st.markdown("---")
        st.markdown("### 💡 Características")
        st.markdown("""
        - ✅ Extracción mejorada de tablas
        - ✅ LLM liviano (Flan-T5)
        - ✅ Vector Database (FAISS)
        - ✅ Búsqueda semántica
        - ✅ Análisis comparativo IA
        """)

    # Contenido principal
    if pagina == "🏠 Inicio":
        st.markdown("""
        ## 🎯 Bienvenido al Sistema RAG de Competencias

        Este sistema utiliza **Inteligencia Artificial avanzada** para analizar documentos académicos:

        ### 🚀 Funcionalidades Principales

        1. **📋 Análisis de PDA**: Detecta competencias en Planes Docentes usando IA
        2. **📊 Análisis de RAE**: Extrae y analiza tablas de Reflexión de Aprendizaje Esperado
        3. **🔍 Búsqueda Vectorial**: Encuentra competencias similares usando embeddings
        4. **⚙️ Configuración**: Gestiona parámetros y estado del sistema

        ### 🔧 Tecnologías Utilizadas

        - **🧠 LLM**: Flan-T5 para análisis inteligente
        - **📊 Vector DB**: FAISS para búsqueda eficiente
        - **🔍 Embeddings**: SentenceTransformers para similitud semántica
        - **📑 Extracción**: pdfplumber para tablas mejoradas

        ### 🎓 Mejoras Implementadas

        - ✅ Extracción de tablas **sin dependencia de Java**
        - ✅ Análisis con **LLM liviano y rápido**
        - ✅ **Vector Database** para búsquedas inteligentes
        - ✅ **Comparación automática** PDA vs RAE
        - ✅ **Interfaz mejorada** con métricas y alertas

        **👈 Selecciona un módulo en la barra lateral para comenzar**
        """)

        # Mostrar métricas del sistema si están disponibles
        if 'vector_db' in st.session_state:
            st.subheader("📊 Resumen del Sistema")
            vector_db = st.session_state.vector_db

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 Documentos", len(vector_db.documents))
            with col2:
                tipos = [meta.get('tipo', 'N/A') for meta in vector_db.metadata]
                st.metric("🏷️ Tipos", len(set(tipos)))
            with col3:
                st.metric("🔍 Dimensiones", vector_db.index.d if vector_db.index else 0)
            with col4:
                st.metric("🚀 Estado", "Activo")

    elif pagina == "📋 Análisis PDA":
        main_pda()

    elif pagina == "📊 Análisis RAE":
        main_reflexion_rae()

    elif pagina == "🔍 Búsqueda Vectorial":
        main_vector_search()

    elif pagina == "⚙️ Configuración":
        main_configuracion()


if __name__ == "__main__":
    main()