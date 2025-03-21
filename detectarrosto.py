import cv2

# Carrega o classificador pré-treinado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carrega a imagem
image = cv2.imread('pessoas.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converte para escala de cinza

# Detecta rostos
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Desenha retângulos ao redor das faces detectadas
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Exibe a imagem com as detecções
cv2.imshow('Faces detectadas', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
