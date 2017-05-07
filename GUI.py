from tkinter import *


def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox('all'))

    


root = Tk()

def submit1():
    cgpa_10s=int(cgpa_10.get())
    cgpa_12s=int(cgpa_12.get())
    res.append(cgpa_10s)
    res.append(cgpa_12s)
    print(res)
    
    
    

canvas = Canvas(root, width=1200)
canvas.pack(side=LEFT)

scrollbar = Scrollbar(root, command=canvas.yview)
scrollbar.pack(side=LEFT, fill='y')

canvas.configure(yscrollcommand = scrollbar.set)

canvas.bind('<Configure>', on_configure)

frame = Frame(canvas,width=1150)
canvas.create_window((0,0), window=frame, anchor='nw')
#frame.pack(fill = BOTH)
var = IntVar()
fo = open("1.txt", "r+")

res = []
class Quest:
    def __init__(self,q):
        self.ques=q
        self.var= IntVar()
    def sel(self):
        res.append((self.var).get())
def prin(obj):
    
    label1 = Label(frame, width=1100)
    var1 = StringVar()
    label1 = Message(frame, textvariable=var1, width= 1000)
    var1.set(obj.ques)
    label1.pack(anchor = W, fill = BOTH)
    Radiobutton(frame, text="Option 1", variable=obj.var, value=1,command=obj.sel).pack( anchor = W )
    Radiobutton(frame, text="Option 2", variable=obj.var, value=2,command=obj.sel).pack( anchor = W )
    Radiobutton(frame, text="Option 3", variable=obj.var, value=3,command=obj.sel).pack( anchor = W )    



labelname = Label(frame)
var2 = StringVar()
labelname = Message(frame, textvariable=var2,width="200")
var2.set("Name")
labelname.pack(side=TOP)
namevar=StringVar()
name=Entry(frame,textvariable=namevar)
name.pack()


labelgen = Label(frame)
vargen = StringVar()
labelgen = Message(frame, textvariable=vargen,width="200")
vargen.set("Gender")
labelgen.pack(side=TOP)
vargender=StringVar()
gender=Entry(frame,textvariable=vargender)
gender.pack()



label10 = Label(frame)
var10 = StringVar()
label10 = Message(frame, textvariable=var10,width="200")
var10.set("X Perc/cgpa")
label10.pack(side=TOP)
var_10=StringVar()
cgpa_10=Entry(frame,textvariable=var_10)
cgpa_10.pack()

label12 = Label(frame)
var12 = StringVar()
label12 = Message(frame, textvariable=var12,width="200")
var12.set("XII Perc/cgpa")
label12.pack(side=TOP)
var_12=StringVar()
cgpa_12=Entry(frame,textvariable=var_12)
cgpa_12.pack()

for i in range(26):
    line = fo.readline()
    q = Quest(line)
    prin(q)    

sub=Button(frame,text="submit",command=lambda:submit1())
sub.pack()
print(res)

root.mainloop()
