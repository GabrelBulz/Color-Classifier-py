from tkinter import *
import tensorflow as tf 

labelList = [
  'red-ish',
  'green-ish',
  'blue-ish',
  'orange-ish',
  'yellow-ish',
  'pink-ish',
  'purple-ish',
  'brown-ish',
  'grey-ish'
]

model = tf.keras.models.load_model('trained_model.h5')

r=0
g=0
b=0

root = Tk()
root.geometry('300x300')

c = Canvas(root, height = 100, width=100, bg='blue')


def update_r(event):
    global r
    r = red.get()
    colorval = "#%02x%02x%02x" % (r, g, b) 
    c.create_rectangle(0,0,102,102,fill=colorval)
    c.pack()
    print_prediction()

def update_g(event):
    global g
    g = green.get()
    colorval = "#%02x%02x%02x" % (r, g, b) 
    c.create_rectangle(0,0,102,102,fill=colorval)
    c.pack()
    print_prediction()

def update_b(event):
    global b
    b = blue.get()
    colorval = "#%02x%02x%02x" % (r, g, b) 
    c.create_rectangle(0,0,102,102,fill=colorval)
    c.pack()
    print_prediction()
   

def print_prediction():
    global r,g,b, model
    tf_input = tf.constant([[r/255,g/255,b/255]])

    result = model.predict(tf_input)
    index = tf.keras.backend.argmax(result, 1).numpy()[0]
    print(labelList[index])



colorval = "#%02x%02x%02x" % (r, g, b) 

c.create_rectangle(0,0,102,102,fill=colorval)
c.pack()

red = Scale(root, from_=0, to=255, orient=HORIZONTAL, length = 200, command=update_r)
red.pack()
green = Scale(root, from_=0, to=255, orient=HORIZONTAL, length = 200, command=update_g)
green.pack()
blue = Scale(root, from_=0, to=255, orient=HORIZONTAL, length = 200, command=update_b)
blue.pack()

# string_predict = StringVar()
# string_predict.set('ana')
# result_predict = Label(root, text=string_predict, font=('Arial', 24))
# result_predict.pack()
# root.update()

root.mainloop()

