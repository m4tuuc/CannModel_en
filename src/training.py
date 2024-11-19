from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd
import torch
import os
import argparse


# recomiendo usar gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  #depura posibles errores en GPU
print("CUDA disponible:", torch.cuda.is_available())



df = pd.read_csv('CannModel/dataset/cannabis.csv')

df = df[df['Type'].isin(['sativa', 'indica', 'hibrida'])]

# Mapear las etiquetas de clase a valores numéricos
label_mapping = {'sativa': 0, 'indica': 1, 'hibrida': 2}
df['labels'] = df['Type'].map(label_mapping)


df = df.dropna(subset=['Description'])

df['Description'] = df['Description'].astype(str)


df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df = df.dropna(subset=['Rating'])


df['Rating'] = df['Rating'].astype(int)

# Guardar el DataFrame limpio en un nuevo archivo CSV
clean_csv_path = 'CannModel/dataset/cannabis_clean.csv'
df.to_csv(clean_csv_path, index=False)

#Cargar el datasetcon Hugging Face datasets
dataset = load_dataset('csv', data_files=clean_csv_path)

# Tokenizar el texto usando el tokenizer de BERT
tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')

def preprocess_function(examples):
    return tokenizer(examples['Description'], padding="max_length", truncation=True)

# Tokenizar el dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Asegurarse de que 'labels' estén en long y las entradas correctas
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Dividir el dataset en entrenamiento y validacion
train_test_split = tokenized_dataset['train'].train_test_split(test_size=0.2, seed=42)  # Añadir semilla para reproducibilidad
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# argumentos de entrenamiento

def parse_args():
    parser = argparse.ArgumentParser(description='Train BERT model for cannabis classification')
    parser.add_argument('--input_file', type=str, default='CannModel/dataset/cannabis_clean.csv',
                      help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default='src/results/final_model',
                      help='Path to save the final model')
    parser.add_argument('--epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                      help='Evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                      help='Learning rate')
    return parser.parse_args()

def train_model(model, train_dataset, eval_dataset, training_args):
    """Función separada para el entrenamiento para mejor manejo de errores"""
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,

            tokenizer=tokenizer
        )
        # Deshabilitar el warning específico
        import warnings
        warnings.filterwarnings("ignore", message=".*Tried to instantiate.*")
        
        return trainer.train()
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        raise


def main():
    args = parse_args()

    
    training_args = TrainingArguments(
        output_dir='./results/checkpoint',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=False,  # Deshabilitar precisión mixta
        dataloader_num_workers=0,  # Reducir workers si hay problemas
        disable_tqdm=False,  # Mantener la barra de progreso
        load_best_model_at_end=True,
    )
    # Cargar el modelo con la capa de clasificación
    model = BertForSequenceClassification.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', num_labels=3,return_dict=True,
        output_attentions=False,
        output_hidden_states=True)
    model.to(device)


    train_model(model, train_dataset, eval_dataset, training_args)
    #inputs = {key: value.to(device) for key, value in inputs.items()}

# Usar el Trainer para entrenar y evaluar
   # trainer = Trainer(
   #     model=model,
   #     args=training_args,
   #     train_dataset=train_dataset,
   #     eval_dataset=eval_dataset,
   #     tokenizer=tokenizer,
   # )

    # Entrenar el modelo
    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)  # Usar save_pretrained en lugar de save_model
    tokenizer.save_pretrained(save_path)


if __name__ == '__main__':
    main()