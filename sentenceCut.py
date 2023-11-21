import tkinter as tk
import re


def on_text_change(event):
    text = text_widget.get("1.0", "end-1c")
    print("입력된 텍스트:", text)


def extract_previous_sentence(event):
    text = text_widget.get("1.0", "end-1c")
    sentences = re.split(r'[.,?!]|\n', text)
    print(sentences)


app = tk.Tk()
app.title("email")

text_widget = tk.Text(app)
text_widget.pack()

text_widget.bind("<KeyRelease>", extract_previous_sentence)

app.mainloop()
