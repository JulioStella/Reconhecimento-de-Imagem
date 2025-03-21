# -*- coding: utf-8 -*-
## RECONHECIMENTO FACIAL EM VIDEOS DO YOUTUBE (MOSTRA IDADE E EMOÇÃO)



import yt_dlp
import os
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont

# URL de um vídeo do YouTube
url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'  # Adicionar o video de preferência

# Nome do arquivo de saída
video_filename = "video.mp4"

# Configuração do yt_dlp para baixar em MP4
ydl_opts = {
    'format': 'bestvideo+bestaudio',
    'merge_output_format': 'mp4',
    'outtmpl': video_filename,
    'quiet': False  # Exibir logs do download
}

# Baixa o vídeo
try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Verifica se o arquivo foi baixado corretamente
    if not os.path.exists(video_filename):
        raise FileNotFoundError("O vídeo não foi baixado corretamente.")

    print(f"Vídeo baixado com sucesso: {video_filename}")

except Exception as e:
    print("Erro ao baixar o vídeo do YouTube:", e)
    exit()

# Carrega o classificador de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Abre o vídeo
cap = cv2.VideoCapture(video_filename)
frame_skip = 5  # Processar um frame a cada 5 para melhor desempenho e precisão
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Sai do loop se não houver mais frames
    
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Pula frames para acelerar
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converte para escala de cinza
    rostos = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=7, minSize=(40, 40))

    # Converte frame para PIL para melhor renderização de texto
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    
    try:
        fonte = ImageFont.truetype("arial.ttf", 24)
    except:
        fonte = ImageFont.load_default()
    
    # Analisa cada rosto detectado
    for (x, y, w, h) in rostos:
        regiao_rosto = frame[y:y+h, x:x+w]
        regiao_rosto = cv2.resize(regiao_rosto, (48, 48))  # Reduz tamanho para melhorar desempenho
        
        try:
            analise = DeepFace.analyze(regiao_rosto, actions=['age', 'emotion'], enforce_detection=False)
            idade = analise[0].get('age', 'Desconhecida')
            emocao = analise[0].get('dominant_emotion', 'Desconhecida')
        except Exception as e:
            idade = "Desconhecida"
            emocao = "Desconhecida"
            print("Erro na análise de idade e emoção:", e)
        
        # Desenha retângulo vermelho ao redor do rosto
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=3)
        draw.text((x, y - 30), f"Idade: {idade}, Emoção: {emocao}", font=fonte, fill=(0, 255, 0))
    
    # Converte de volta para OpenCV
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    
    # Exibe o frame com taxa de atualização melhorada
    cv2.imshow('Detecção de Rosto', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
