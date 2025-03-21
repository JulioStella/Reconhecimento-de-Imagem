# -*- coding: utf-8 -*-

## FAZ DETECÇÃO FACIAL DE FOTO COM EMOÇÃO E IDADE
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont

# Carrega o classificador pré-treinado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carrega a imagem
imagem = cv2.imread('pessoas.jpg')
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)  # Converte para escala de cinza

# Detecta rostos
rostos = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Desenha os retângulos nos rostos diretamente na imagem OpenCV
for (x, y, w, h) in rostos:
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Vermelho

# Converte imagem para formato PIL
imagem_pil = Image.fromarray(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(imagem_pil)

# Carrega uma fonte compatível
try:
    fonte = ImageFont.truetype("arial.ttf", 32)
except:
    fonte = ImageFont.load_default()

# Analisa cada rosto detectado e adiciona texto
for (x, y, w, h) in rostos:
    regiao_rosto = imagem[y:y+h, x:x+w]  # Recorta a região do rosto
    
    try:
        analise = DeepFace.analyze(regiao_rosto, actions=['age', 'emotion'], enforce_detection=False)
        idade = analise[0]['age']
        emocao = analise[0]['dominant_emotion']
    except Exception as e:
        idade = "Desconhecida"
        emocao = "Desconhecida"
        print("Erro na análise de idade e emoção:", e)
    
    # Adiciona texto usando PIL
    draw.text((x, y - 40), f"Idade: {idade}, Emoção: {emocao}", font=fonte, fill=(0, 255, 0))

# Converte de volta para OpenCV
imagem = cv2.cvtColor(np.array(imagem_pil), cv2.COLOR_RGB2BGR)

# Exibe a imagem com as detecções
cv2.imshow('Rostos detectados com idade e emoção', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
