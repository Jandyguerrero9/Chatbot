# training.py
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Cargar el modelo preentrenado y el tokenizador
model_name = 'gpt2'  # O puedes usar cualquier otro modelo basado en transformers
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Establecer el token de padding
tokenizer.pad_token = tokenizer.eos_token  # Usamos el token de fin de secuencia (eos_token) como el pad_token

# Cargar el dataset desde un archivo CSV
dataset_path = '/content/drive/MyDrive/chatbot/hipolito_mejia_dataset.csv'  # Ajusta la ruta según sea necesario
df = pd.read_csv(dataset_path)

# Verificar las primeras filas para asegurarse de que las columnas son correctas
print(df.head())

# Concatenar las columnas 'Pregunta' y 'Respuesta' para crear un prompt completo
# Ejemplo de un prompt: "Pregunta: ¿Cuál es el nombre de Hipólito Mejía? Respuesta: Hipólito Mejía."
df['text'] = "pregunta: " + df['pregunta'] + " respuesta: " + df['respuesta']

# Seleccionar la columna de texto combinada
df = df[['text']]

# Convertir el dataframe a un formato que Hugging Face pueda usar
dataset = Dataset.from_pandas(df)

def preprocess_function(examples):
    encodings = tokenizer(
        examples['text'],
        truncation=True,
        padding="max_length",
        max_length=256,  # Ajustar según la longitud promedio
        return_tensors='pt'
    )
    encodings['labels'] = encodings['input_ids'].clone()
    return encodings


tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Dividir en conjunto de entrenamiento y validación
train_dataset = tokenized_datasets
eval_dataset = tokenized_datasets  # Si tienes un conjunto de validación separado, ajústalo

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    eval_steps=500,  # Evaluar cada 500 pasos
    logging_steps=100,  # Registrar cada 100 pasos
    save_steps=1000,  # Guardar un checkpoint cada 1000 pasos
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,  # Más épocas para aprovechar mejor los datos
    weight_decay=0.01,
    warmup_steps=500,  # Warmup para un aprendizaje más estable
    fp16=True,  # Activar precisión mixta si es compatible
    save_total_limit=2,  # Mantener solo los 2 últimos checkpoints
)
# Inicializar el trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo entrenado
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
