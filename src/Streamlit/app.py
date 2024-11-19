import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import numpy as np
import plotly.graph_objects as go
from collections import Counter
import re

# Configurar la pag
st.set_page_config(
    page_title="Cannabis Recommendation System",
    page_icon="🌿",
    layout="wide"
)

# Cargar modelo y tokenizer.
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model_path = 'CannModel/src/results/checkpoint/...'
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None, None


@st.cache_data  # Cache para los datos
def load_data():
    try:
        df = pd.read_csv('CannModel/src/Streamlit/dataset/cannabis_clean.csv')
        # Normalizar nombres de columnas: minusculas y sin espacios
        df.columns = df.columns.str.strip().str.lower()
        
        if df.empty:
            st.error("El archivo de datos está vacío")
            return None
        return df
    except Exception as e:
        st.error(f"Error cargando los datos: {str(e)}")
        return None

def get_embedding(df, text, model, tokenizer):
    if df is None:
        st.error("No hay datos para generar embeddings")
        return None
        
    try:
        # Asegurar modo evaluación
        model.eval()
        
        # Tokenizar y preparar input
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            # Obtener las salidas del modelo base (bert)
            outputs = model.bert(**inputs)
            
            # Obtener el embedding del token [CLS] (primer token)
            # Shape: [1, 768]
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Asegurarse de que tenemos la dimensión correcta
            if len(embeddings.shape) == 2:
                return embeddings.squeeze()
            
            # Dentro de get_embedding, antes del return:
            st.write("Debug - Dimensiones:")
            st.write(f"Input shape: {inputs['input_ids'].shape}")
            st.write(f"Output shape: {embeddings.shape}")
            
            return embeddings
            
    except Exception as e:
        st.error(f"Error generando embeddings: {str(e)}")
        st.error(f"Forma del tensor: {embeddings.shape if 'embeddings' in locals() else 'No disponible'}")
        return None

def get_recommendations(df, text, model, tokenizer):
    if model is None or tokenizer is None:
        st.error("No se pudo cargar el modelo o tokenizer")
        return None

    # Debug para mprimir las columnas del DataFrame
    #st.write("Debug - Columnas del DataFrame:", df.columns.tolist())

    try:
        # Generar embedding para la descripcion del usuario
        user_embedding = get_embedding(df, text, model, tokenizer)
        if user_embedding is None:
            return None
        
        # Generar embeddings para todas las descripciones en el DataFrame
        strain_embeddings = []
        for description in df['description']:
            embedding = get_embedding(df, description, model, tokenizer)
            strain_embeddings.append(embedding)
        
        # Convertir a array de numpy para cálculos más eficientes
        strain_embeddings = np.array(strain_embeddings)
        
        # Calcular similitud coseno entre el embedding del usuario y los embeddings de las cepas
        similarities = np.dot(strain_embeddings, user_embedding) / (
            np.linalg.norm(strain_embeddings, axis=1) * np.linalg.norm(user_embedding)
        )
        
        # Encontrar la cepa más similar
        closest_index = np.argmax(similarities)
        confidence_percentage = similarities[closest_index]
        
        # Obtener la cepa recomendada
        closest_strain = df.iloc[closest_index]

        #Se agrego una lógica para separar palabras concatenadas usando expresiones regulares(re.findall)
        # Manejo de effects
        effects = []
        if 'effects' in closest_strain and pd.notna(closest_strain['effects']):
            effect_value = str(closest_strain['effects'])
            try:
                # Remover corchetes si existen
                if effect_value.startswith('[') and effect_value.endswith(']'):
                    effect_value = effect_value[1:-1]
                # Dividir por comas y limpiar
                raw_effects = [e.strip().strip('"\'') for e in effect_value.split(',') if e.strip()]
                # Separar efectos concatenados
                for effect in raw_effects:
                    # Expresion para dividir por mayúsculas
                    split_effects = re.findall('[A-Z][^A-Z]*', effect)
                    effects.extend(split_effects)
            except Exception as e:
                st.error(f"Error procesando effects: {str(e)}")
                effects = [effect_value]

        # Manejo similar para flavors
        flavors = []
        if 'flavor' in closest_strain and pd.notna(closest_strain['flavor']):
            flavor_value = str(closest_strain['flavor'])
            try:
                if flavor_value.startswith('[') and flavor_value.endswith(']'):
                    flavor_value = flavor_value[1:-1]
                raw_flavors = [f.strip().strip('"\'') for f in flavor_value.split(',') if f.strip()]
                for flavor in raw_flavors:
                    #Mismo aqui
                    split_flavors = re.findall('[A-Z][^A-Z]*', flavor)
                    flavors.extend(split_flavors)
            except Exception as e:
                st.error(f"Error procesando flavors: {str(e)}")
                flavors = [flavor_value]
        
        recommendations = {
            'strain_type': str(closest_strain['type']),
            'confidence_percentage': round(float(confidence_percentage) * 100, 2),
            'strain_description': str(closest_strain['description']),
            'effects': effects,
            'flavors': flavors
        }
        
        # Debug: Imprimir el diccionario final
        #st.write("Debug - Recommendations dict:", recommendations)
        
        return recommendations
    except Exception as e:
        st.error(f"Error al procesar la recomendación: {e}")
        st.error(f"Detalles del error: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def main():         

    model, tokenizer = load_model_and_tokenizer()
    df = load_data()
    
    if model is None or tokenizer is None or df is None:
        st.error("No se pudieron cargar los recursos necesarios")
        return

    # Título y descripción
    st.title("🌿 Cannabis Recommendation System")
    st.markdown("### Find your perfect cannabis strain")

    # Entrada
    description = st.text_area(
        "Describe your needs:",
        placeholder="Example: I'm looking for something relaxing and aromatic to relieve stress",
        height=100
    )

    # Botón de predicción
    if st.button("Get Recommendation", type="primary"):
        if description:
            with st.spinner('Processing your request...'):
                result = get_recommendations(
                    text=description,
                    df=df,
                    model=model,
                    tokenizer=tokenizer
                )
                
                if result:
                    # Mostrar resultados
                    st.success(f"🥬 Recommended Strain: {result['strain_type']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Confidence Level", f"{result['confidence_percentage']}%")
    
                    st.subheader("📝 Description")
                    st.info(result['strain_description'])

                    # Columnas para efectos y sabores
                    col3, col4 = st.columns(2)
                    with col3:
                        st.subheader("✨ Effects")
                        if isinstance(result['effects'], str):
                            effects = []
                            current_word = ""
                            for char in result['effects']:
                                if char.isupper() and current_word:
                                    effects.append(current_word)
                                    current_word = char
                                else:
                                    current_word += char
                            if current_word:
                                effects.append(current_word)
                        else:
                            effects = result['effects']
                        
                        # Crear gráfico de pie para efectos
                        if effects:
                            effect_counts = Counter(effects)
                            fig_effects = go.Figure(data=[go.Pie(
                                labels=list(effect_counts.keys()),
                                values=list(effect_counts.values()),
                                hole=.3,
                                marker=dict(colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC'])
                            )])
                            fig_effects.update_layout(
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                margin=dict(t=0, l=0, r=0, b=0),
                                height=300
                            )
                            st.plotly_chart(fig_effects, use_container_width=True)

                    with col4:
                        st.subheader("🌺 Flavors")
                        if isinstance(result['flavors'], str):
                            flavors = []
                            current_word = ""
                            for char in result['flavors']:
                                if char.isupper() and current_word:
                                    flavors.append(current_word)
                                    current_word = char
                                else:
                                    current_word += char
                            if current_word:
                                flavors.append(current_word)
                        else:
                            flavors = result['flavors']
                        
                        # Crear gráfico de pie para sabores
                        if flavors:
                            flavor_counts = Counter(flavors)
                            fig_flavors = go.Figure(data=[go.Pie(
                                labels=list(flavor_counts.keys()),
                                values=list(flavor_counts.values()),
                                hole=.3,
                                marker=dict(colors=['#FFB366', '#99FF99', '#FF99FF', '#99CCFF', '#FFFF99'])
                            )])
                            fig_flavors.update_layout(
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                margin=dict(t=0, l=0, r=0, b=0),
                                height=300
                            )
                            st.plotly_chart(fig_flavors, use_container_width=True)

        else:
            st.warning("⚠️ Por favor, ingresa una descripción")

    with st.expander("ℹ️ Information about strain types"):
        st.write("""
        - **Indica**: Known for its relaxing and calming effects.
        - **Sativa**: Provides energizing and stimulating effects.
        - **Hybrid**: Combines characteristics of both varieties.
        """)

if __name__ == "__main__":
    main()
