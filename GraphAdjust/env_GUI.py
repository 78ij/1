import env_subgraph
import trimesh
import pickle
import numpy as np
import matplotlib.pyplot as plt
#Import the tkinter library
from tkinter import *
import tkinter.messagebox
import numpy as np
import cv2
import os
from PIL import Image, ImageTk
import time

class Application(Frame):
    def __init__(self, master=None):
        super().__init__(master)

        self.playing = False
        master.bind("<Up>",self.up)
        master.bind("<Down>",self.down)
        master.bind("<Right>",self.right)
        master.bind("<Left>",self.left)
        master.bind("<Prior>",self.prev)
        master.bind("<Next>",self.next)
        master.bind("<Control-z>",self.undo)

        with open('RL_train_data_ldroom_complete_subbox_2.pkl','rb') as f:
            train_data = pickle.load(f)
        train_data_size = len(train_data)

        self.env_list = []
        list2 = []
        for i in range(1, train_data_size):
            print(i)
            try:
                env_tmp = env_subgraph.ENV(train_data[i])
                coll, _ = env_tmp.getboxcollision()
                if coll != 0:
                    self.env_list.append(env_tmp)
                    if len(self.env_list) == 100: break
            except:
                continue
        self.env_list = [self.env_list[75]]
        print(list2)
        self.master = master
        self.grid(row=0,column=0)
        self.create_widgets()

    def create_widgets(self):
    
        #Create a Label to display the image   
        self.nextbtn = Button(self, text="NEXT SCENE", fg="black",
                              command=self.nextscene)
        self.nextbtn.grid(row=0,column=2)

        self.prevbtn = Button(self, text="PREV SCENE", fg="black",
                              command=self.prevscene)
        self.prevbtn.grid(row=0,column=0)

        self.savebtn = Button(self, text="SAVE", fg="black",
                              command=self.save)
        self.savebtn.grid(row=0,column=1)
        # self.prev = Button(self, text="QUIT", fg="red",
        #                       command=self.master.destroy)
        # self.prev.pack(side="bottom")
        self.action_array = []

        self.current_item_id = 0
        self.current_env_idx = 0

        self.X = self.env_list[self.current_env_idx].visualize2D_GUI(highlight_idx = self.current_item_id)
        img_cvt = cv2.cvtColor(self.X, cv2.COLOR_RGBA2BGR)
        blue,green,red = cv2.split(img_cvt)
        img = cv2.merge((red,green,blue))
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        self.panel = Label(self, image=imgtk)        
        self.panel.photo = imgtk
        self.panel.grid(row=3,column=0,columnspan=3)

        self.playbtn = Button(self, text="PLAY", fg="black",
                              command=self.play)
        self.playbtn.grid(row=2,column=0)
        self.loadbtn = Button(self, text="LOAD", fg="black",
                              command=self.load)
        self.loadbtn.grid(row=2,column=2)

        self.list = Listbox(self,width=70,yscrollcommand=True,selectmode='single')
        self.list.grid(row=1,column=0,columnspan=3)

    def nextscene(self):
        if self.playing: return
        if(self.current_env_idx == len(self.env_list) -1): return
        self.list.selection_clear(0,last=END)
        self.list.delete(first=0,last=END)
        self.current_item_id = 0
        self.action_array = []
        self.current_env_idx += 1
        self.env_list[self.current_env_idx].reset()
        print(self.env_list[self.current_env_idx].item_count_real)
        self.update_image()
    def prevscene(self):
        if self.playing: return
        if(self.current_env_idx == 0): return
        self.list.selection_clear(0,last=END)
        self.list.delete(first=0,last=END)
        self.current_item_id = 0
        self.action_array = []
        self.current_env_idx -= 1
        self.env_list[self.current_env_idx].reset()
        self.update_image()

    def next(self,event):
        if self.playing: return
        if(self.current_item_id == self.env_list[self.current_env_idx].item_count_real -1): return
        self.current_item_id += 1
        self.update_image()

    def save(self):
        result = 1
        if os.path.exists('./saved_operations/' + str(self.current_env_idx) + '.action'):
            result = tkinter.messagebox.askokcancel("Overwrite","There is an existing action file of env index " + str(self.current_env_idx) + ". Overwrite?")
        if result == 1:
            with open('./saved_operations/' + str(self.current_env_idx) + '.action','wb') as f:
                pickle.dump(self.action_array,f)
    def load(self):
        result = 1
        if not os.path.exists('./saved_operations/' + str(self.current_env_idx) + '.action'):
            result = tkinter.messagebox.showerror("No file exist!","There is no existing action file of env index " + str(self.current_env_idx) + ".")
            return
        else:
            result = tkinter.messagebox.askokcancel("Reset!","loading an action array will result in the reset of your current actions! Continue?")
            if result == 0: return
            with open('./saved_operations/' + str(self.current_env_idx) + '.action','rb') as f:
                self.action_array = pickle.load(f)
            self.env_list[self.current_env_idx].reset()
            self.list.delete(first=0,last=END)
            self.list.select_clear(0, last=END)
            for i in self.action_array:
                item = int(i / 4)
                action = i % 4
                r,d = self.env_list[self.current_env_idx].step(i)
                if action == 0:
                    self.list.insert('end','Moved item id ' + str(item) + ' towards RIGHT direction, get reward = ' + str(r) + '.')
                if action == 1:
                    self.list.insert('end','Moved item id ' + str(item) + ' towards LEFT direction, get reward = ' + str(r) + '.')
                if action == 2:
                    self.list.insert('end','Moved item id ' + str(item) + ' towards UP direction, get reward = ' + str(r) + '.')
                if action == 3:
                    self.list.insert('end','Moved item id ' + str(item) + ' towards DOWN direction, get reward = ' + str(r) + '.')
            self.update_image()


    def prev(self,event):
        if self.playing: return
        if(self.current_item_id == 0): return
        self.current_item_id -= 1
        self.update_image()

    def undo(self,event):
        if self.playing: return
        if(len(self.action_array) == 0): return
        self.list.delete(first=END)
        self.list.select_set(first=END)
        last_action = self.action_array[-1]
        item = int(last_action / 4)
        action = last_action % 4
        if action == 0: newaction = 1
        if action == 1: newaction = 0
        if action == 2: newaction = 3
        if action == 3: newaction = 2
        self.env_list[self.current_env_idx].step(item * 4 + newaction)
        self.action_array = self.action_array[:-1]
        self.update_image()

    def play(self):
        self.playing = True
        self.list.select_clear(0, last=END)
        self.list.select_set(first=0)
        array_tmp = []
        for i in range(len(self.action_array) - 1,-1,-1):
            last_action = self.action_array[-1]
            item = int(last_action / 4)
            action = last_action % 4
            if action == 0: newaction = 1
            if action == 1: newaction = 0
            if action == 2: newaction = 3
            if action == 3: newaction = 2
            self.env_list[self.current_env_idx].step(item * 4 + newaction)
            array_tmp.insert(0,last_action)
            self.action_array = self.action_array[:-1]
        self.update_image()
        self.update()
        time.sleep(1)
        for i in range(len(array_tmp)):
            self.list.select_clear(0, last=END)
            self.list.select_set(first=i)
            self.action_array.append(array_tmp[i])
            self.env_list[self.current_env_idx].step(array_tmp[i])
            self.update_image()
            self.update()
            time.sleep(1)
        self.playing = False
    def up(self,event):
        if self.playing: return
        self.list.select_clear(0, last=END)
        self.list.select_set(first=END)
        self.action_array.append(self.current_item_id * 4 + 2)
        r,d = self.env_list[self.current_env_idx].step(self.current_item_id * 4 + 2)
        self.list.insert('end','Moved item id ' + str(self.current_item_id) + ' towards UP direction, get reward = ' + str(r) + '.')
        self.update_image()
    def down(self,event):
        if self.playing: return
        
        self.list.select_clear(0, last=END)
        self.list.select_set(first=END)
        self.action_array.append(self.current_item_id * 4 + 3)
        r,d = self.env_list[self.current_env_idx].step(self.current_item_id * 4 + 3)
        self.list.insert('end','Moved item id ' + str(self.current_item_id) + ' towards DOWN direction, get reward = ' + str(r) + '.')
        self.update_image()
    def right(self,event):
        if self.playing: return
        
        self.list.select_clear(0, last=END)
        self.list.select_set(first=END)
        self.action_array.append(self.current_item_id * 4 + 0)
        r,d = self.env_list[self.current_env_idx].step(self.current_item_id * 4 + 0)
        self.list.insert('end','Moved item id ' + str(self.current_item_id) + ' towards RIGHT direction, get reward = ' + str(r) + '.')
        self.update_image()
    def left(self,event):
        if self.playing: return
        
        self.list.select_clear(0, last=END)
        self.list.select_set(first=END)
        self.action_array.append(self.current_item_id * 4 + 1)
        r,d = self.env_list[self.current_env_idx].step(self.current_item_id * 4 + 1)
        self.list.insert('end','Moved item id ' + str(self.current_item_id) + ' towards LEFT direction, get reward = ' + str(r) + '.')
        self.update_image()
    def update_image(self):
        self.X = self.env_list[self.current_env_idx].visualize2D_GUI(highlight_idx = self.current_item_id)
        img_cvt = cv2.cvtColor(self.X, cv2.COLOR_RGBA2BGR)
        blue,green,red = cv2.split(img_cvt)
        img = cv2.merge((red,green,blue))
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        self.panel.configure(image=imgtk)
        self.panel.photo = imgtk

root = Tk()
app = Application(master=root)

app.mainloop()
