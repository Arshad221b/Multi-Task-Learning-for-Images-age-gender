import cv2
import torch
from torch import nn
from torchvision import transforms, models, datasets

PATH = '/Users/arshad_221b/Downloads/Projects/multi-task-learning-for-age-and-gender-/state_dict_model.pt'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
IMAGE_SIZE = 224
def preprocess_image(im):
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])

    im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
    im = torch.tensor(im).permute(2,0,1)
    im = transform(im/255.)
    return im[None]


def get_model():
    model = models.vgg16(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.Sequential(
        nn.Conv2d(512,512, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )
    class ageGenderClassifier(nn.Module):
        def __init__(self):
            super(ageGenderClassifier, self).__init__()
            self.intermediate = nn.Sequential(
                nn.Linear(2048,512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512,128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128,64),
                nn.ReLU(),
            )
            self.age_classifier = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            self.gender_classifier = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            x = self.intermediate(x)
            age = self.age_classifier(x)
            gender = self.gender_classifier(x)
            return gender, age
        
    model.classifier = ageGenderClassifier()
    return model

model = get_model()
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        face = preprocess_image(face)
        gen, age = model(face)

        print(gen, age.item())
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        gen = "Female" if gen.item() > 0.5 else "Male"
        age = str(int(age.item() * 80))
        cv2.putText(img, age, (700, 300), cv2.FONT_HERSHEY_COMPLEX, 4.0, (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.putText(img, gen, (700, 200), cv2.FONT_HERSHEY_COMPLEX, 4.0, (255, 255, 255), lineType=cv2.LINE_AA)
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
