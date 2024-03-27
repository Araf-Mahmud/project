from customtkinter import *
from PIL import Image

from predict import *

app = CTk()
app.geometry("900x700")

set_appearance_mode("light")

image_file = '../data/test/Bike/Bike (1070).jpeg'

def selectImage():
    filename=filedialog.askopenfilename()
    print(filename)
    global image_file
    image_file=filename
    img=Image.open(filename)
    image=CTkImage(light_image=img,dark_image=img,size=(600,400))
    imLabel=CTkLabel(app,text="",image=image)
    imLabel.place(relx = 0.5, rely = 0.5,anchor="center")



def classify():
    model = CarBikeClassifier(num_classes=2)
    model.load_state_dict(torch.load('../pretrained_models/model.pth', map_location=torch.device('cpu')))
    print(image_file)
    prediction_text=predict_image(model, image_file, device='cpu') 
    print(prediction_text)
    if prediction_text == 'car':
        prediction_text = '✅ Car 🚗'
    else:
        prediction_text = '✅ Bike 🏍'

    frame=CTkFrame(master=app,fg_color="transparent")
    frame.place(relx = 0.5, rely = 0.1,anchor="center")
    txt=CTkLabel(master=frame,text=prediction_text,font=("Roboto",75),text_color="#1b4332",bg_color="#d8f3dc",pady=5,padx=5)
    txt.pack(anchor="s",expand=True,pady=3,padx=3)



select_button = CTkButton(master=app, text = "Select Image", fg_color = "#52b69a", text_color= "white", command = selectImage)
select_button.pack(padx = 5, pady = 5)
select_button.place(relx = 0.4, rely = 0.9,anchor="center")

classify_button=CTkButton(master=app,text="Show Result",fg_color="#7a7a7a", text_color= "white",command=classify)
classify_button.place(relx=0.6,rely=0.9,anchor="center")

app.mainloop()


