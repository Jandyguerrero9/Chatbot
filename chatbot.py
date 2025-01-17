import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random
import flet as ft
import time
import re

class ChatbotApp:
    def __init__(self, model_path):
        # Initialize the model and tokenizer
        self.model_name = model_path
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Enable GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Predefined farewell messages
        self.despedidas = [
            "¡Hasta luego!",
            "¡Adiós, que tengas un buen día!",
            "Nos vemos pronto, ¡cuídate!",
            "Fue un placer ayudarte, ¡hasta la próxima!",
            "¡Hasta la vista!",
            "¡Chao, chao!",
        ]

    def generate_response(self, input_text):
        try:
            # Tokenize the input
            inputs = self.tokenizer(input_text, return_tensors='pt')
            
            # Move input tensors to the same device as the model
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # Generate the response
            output = self.model.generate(
                inputs['input_ids'],
                max_length=150,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode and clean the response
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response = self.clean_response(response, input_text)
            
            return response
        except Exception as e:
            return f"Lo siento, ha ocurrido un error: {str(e)}"

    def clean_response(self, response, input_text):
        # Remove the original input from the response
        response = response.replace(input_text, '').strip()
        
        # Remove any repeated or nonsensical text
        response = re.sub(r'^(Pregunta:|Respuesta:)', '', response).strip()
        
        # Ensure the response starts with a capital letter and ends with punctuation
        if response and not response[0].isupper():
            response = response[0].upper() + response[1:]
        
        if response and not response[-1] in '.!?':
            response += '.'
        
        # Limit response length
        response = response[:300]
        
        return response

def main(page: ft.Page):
    # Configure page
    page.title = "Chatbot de IA"
    page.bgcolor = ft.colors.WHITE
    page.padding = 20
    page.theme_mode = ft.ThemeMode.LIGHT
    
    # Initialize the chatbot
    chatbot = ChatbotApp('/Users/jandymercedes/Downloads/chatbot/model')
    
    # Create UI components
    chat_box = ft.Column(
        scroll=ft.ScrollMode.ADAPTIVE, 
        spacing=10, 
        expand=True, 
        width=600,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH
    )
    
    user_input = ft.TextField(
        label="Escribe tu mensaje", 
        autofocus=True, 
        expand=True,
        border_radius=10,
        border_color=ft.colors.BLUE,
        on_submit=lambda e: send_message(e)
    )
    
    send_button = ft.IconButton(
        icon=ft.icons.SEND_ROUNDED,
        icon_color=ft.colors.BLUE,
        on_click=lambda e: send_message(e)
    )
    
    # Create input row
    input_row = ft.Row([
        user_input,
        send_button
    ], spacing=10, width=600)
    
    # Add title
    page.add(
        ft.Text(
            "Chatbot de IA", 
            size=32, 
            color=ft.colors.BLUE, 
            weight=ft.FontWeight.BOLD,
            text_align=ft.TextAlign.CENTER
        ),
        ft.Divider(),
        chat_box,
        input_row
    )
    
    def send_message(e):
        user_text = user_input.value.strip()
        
        if not user_text:
            return
        
        # Check for exit commands
        if user_text.lower() in ['salir', 'exit', 'adiós']:
            despedida = random.choice(chatbot.despedidas)
            add_message("Tú", user_text, ft.colors.BLACK87)
            add_message("Chatbot", despedida, ft.colors.GREEN)
            user_input.value = ""
            return
        
        # Show user message
        add_message("Tú", user_text, ft.colors.BLACK87)
        
        # Show loading indicator
        loading = ft.Text("Escribiendo...", color=ft.colors.GREY)
        chat_box.controls.append(loading)
        chat_box.update()
        
        # Generate response
        try:
            response = chatbot.generate_response(f"Pregunta: {user_text} Respuesta:")
            
            # Remove loading indicator
            chat_box.controls.remove(loading)
            
            # Add bot response
            add_message("Chatbot", response, ft.colors.BLUE)
        except Exception as ex:
            # Remove loading indicator
            chat_box.controls.remove(loading)
            add_message("Chatbot", f"Lo siento, ha ocurrido un error: {str(ex)}", ft.colors.RED)
        
        # Clear input
        user_input.value = ""
        page.update()
    
    def add_message(sender, message, color):
        chat_box.controls.append(
            ft.Container(
                content=ft.Column([
                    ft.Text(f"{sender}: {message}", color=color)
                ]),
                padding=10,
                border_radius=10,
                bgcolor=ft.colors.BLUE_GREY_50,
                margin=5
            )
        )
        chat_box.update()
        page.update()

# Run the Flet app
ft.app(target=main)