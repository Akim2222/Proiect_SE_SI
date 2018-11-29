import numpy as np
import cv2
import pickle
# incarcam clasificatorii de tip haar respectiv LBP 
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml') # clasificatorul haar - recunoastere faciala
recognizer = cv2.face.LBPHFaceRecognizer_create() # clasificatorul LBP 
recognizer.read("trainer.yml") # Datele antrenate rezultate din antrenarea retelei - identificare faciala

# atribuire de tip value:key pentru vectorii imaginilor si cheia atribuita acestora
labels = {}
with open("labels.pickle", 'rb') as f:
	original_labels = pickle.load(f)
	labels = {v:k for k,v in original_labels.items()}

cap = cv2.VideoCapture(0) # initializare captura video

while(True):
	ret, img = cap.read() # captura cadre 
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convertirea imagini
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) # folosim KNN K nearest neighbors 

# specificarea unei zone de interes in functie de x,y = punctul de start w = lungime si h = inaltime
	for (x,y,w,h) in faces:
		print (x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
# in acest punct, reteaua ofera un scor de incredere prin care se atribuie fetei recunoscute un id_ pentru a identifica fata gasita in imagine
		id_, conf = recognizer.predict(roi_gray)
		if conf >= 45 and conf <=85:
			print(conf)
			print(id_)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			stroke = 2
			color = (255, 255, 255)
# afiseaza id-ul fetei recunoscute, daca scorul de incredere este cuprins intre 45 si 85
			cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
# evidentiaza zona de interes print-o forma rectangulara 
		color = (255, 0, 0)
		stroke = 4
		end_cord_x = x + w
		end_cord_y = y + h
		cv2.rectangle(img, (x, y), (end_cord_x, end_cord_y), color, stroke)

		
		
# afisam rezultatul final al progesarii pe imaginea RGB
	cv2.imshow("window", img)
# conditie care trebuie indeplinita pentru a iesi din bucla infinita.
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()
cap.release()		
