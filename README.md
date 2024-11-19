# Cannabis Strain Recommender 🌿

Un sistema de recomendación basado en BERT para sugerir variedades de cannabis (cepas) según una descripción proporcionada. La aplicación está desarrollada con **Streamlit**. 

basado en https://www.kaggle.com/code/kabure/cannabis-species-eda-and-models-pipeline

**NOTA: ESTE MODELO TODAVIA ESTA EN DESARROLLO Y TARDA UN ESTIMADO DE 1min.30seg EN DAR LA RESPUESTA, CUALQUIER CONTRIBUCION PARA EL DESARROLLO DEL MISMO ES BIENVENIDO.**

## Características

- **Predicción de cepas**: Recomienda variedades de cannabis basadas en descripciones de efectos deseados, sabores o usos.
- **Información detallada**: Devuelve información adicional, como efectos, sabores y usos recomendados.

---


## Requisitos

`pip install -r requirements.txt`


---

## 1.Instalación

**Clona el repositorio**:
git clone https://github.com/m4tuuc/CannModel_en.git

## Crea un entorno virtual
```bash 
python -m venv venv
source venv/bin/activate   # En Linux/Mac
venv\Scripts\activate 
```

**1a. Entrena el modelo
```bash 
python training.py 
```

o puedes bajarlo de aqui:
https://drive.google.com/drive/folders/1aexXKnqbECnIBJYtfRuynm1PZmHHrEdg?usp=sharing

---

## 2.Ejecuta la aplicación Streamlit:

`streamlit run app.py` 
Abre tu navegador e ingresa a http://localhost:8501.



## Uso del modelo 
El modelo toma como entrada una descripción proporcionada por el usuario (por ejemplo, "relaxed and chill") y devuelve la cepa mas adecuada, junto con la confianza de la predicción.

Ejemplo de entrada:
```bash
{
  "description": "relaxed and chill."
}
```
![image](https://github.com/user-attachments/assets/ed2a645a-57a7-4305-98f5-e63758e78030)

Ejemplo de salida:

![image](https://github.com/user-attachments/assets/4643bd39-bb59-47e4-8dce-57becace3630)


---


Las contribuciones son bienvenidas. Si encuentras algun problema o deseas proponer una mejora, no dudes en abrir un issue o un pull request.
