import torch
import os
import sys
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

print(torch.cuda.is_available())
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
os.environ["OMP_NUM_THREADS"] = "1"

# Definir el dispositivo para mover los tensores (GPU si está disponible)
device = torch.device("cuda")

# Ruta del modelo ya entrenado
model_dir = 'CannModel/src/results/your model'

# Verificar si el modelo ya ha sido entrenado y guardado
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"No se encontró el directorio del modelo: {model_dir}. Asegúrate de que el modelo esté entrenado y guardado.")


# Cargar el modelo y el tokenizador ya entrenados
print("Cargando el modelo y el tokenizador entrenados...")
model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
print("Modelo cargado con éxito.")
tokenizer = BertTokenizer.from_pretrained('CannModel/src/results/your model')
print("Tokenizador cargado con éxito.")

test_dataset = load_dataset('csv', data_files='CannModel/dataset/cannabis_clean.csv')['train']

# Texto de entrada para predecir
text = "can assist with inflammation irritability and minor physical discomfort The uplifting clearheaded buzz and soothing physical effects make Aliens"

# Tokenizar el texto
#inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#test_dataset = test_dataset.map(lambda x: tokenizer(x['Description'], padding='max_length', truncation=True), batched=True)
#test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

from torch.utils.data import DataLoader
def preprocess_function(examples):
    return tokenizer(examples['Description'], padding='max_length', truncation=True, max_length=512)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Crear un DataLoader para el dataset de prueba
test_dataloader = DataLoader(tokenized_test_dataset, batch_size=8)

#inputs = {k: v.to(device) for k, v in inputs.items()}

model.eval()
predictions, labels = [], []
with torch.no_grad():
    for batch in test_dataloader:
        # Mover el batch al dispositivo
        batch = {k: v.to(device) for k, v in batch.items()}

        # Obtener las salidas del modelo
        outputs = model(**batch)
        logits = outputs.logits
        preds = logits.argmax(dim=-1)

        # Almacenar las predicciones y etiquetas verdaderas
        predictions.extend(preds.cpu().numpy())
        labels.extend(batch['labels'].cpu().numpy())

# métricas
accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, average='macro')

print(f"Precisión en el conjunto de prueba: {accuracy}")
print(f"F1-Score en el conjunto de prueba: {f1}")
print(torch.cuda.is_available())
