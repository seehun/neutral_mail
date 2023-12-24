
from tkinter import *
import smtplib
import re

# Main Screen Init
master = Tk()
master.title = 'Custom Email App'

# Functions


def apply_background_color(state):
    if state == 'positive':
        return 'olive'
    elif state == 'negative':
        return 'coral'
    else:
        return 'white'


def update_background_color():
    current_color = bodyEntry.cget('bg')
    target_color = apply_background_color(state_var.get())

    current_rgb = master.winfo_rgb(current_color)
    target_rgb = master.winfo_rgb(target_color)

    step = 20
    new_rgb = tuple(
        int(current + step * ((target - current) / 255)) for current, target in zip(current_rgb, target_rgb)
    )

    new_color = "#{:02x}{:02x}{:02x}".format(*new_rgb)
    bodyEntry.configure(bg=new_color)

    master.after(100, update_background_color)


def predict(a):
    state = 'neutral'
    for sentence in a:
        if sentence in ['잘부탁해요', '챙겨주세요', '봐주세요', '잘 좀 부탁드립니다']:
            state = 'positive'
            break
        elif sentence in ['머하자는건지', '잘 좀합시다']:
            state = 'negative'
            break
        else:
            state = 'neutral'
    # print(state)
    # bodyEntry.tag_configure(
    #     "colored", background=apply_background_color(state))
    # bodyEntry.tag_add("colored", "1.0", "end")

    state_var.set(state)
    bodyEntry.tag_configure(
        "colored", background=apply_background_color(state))
    bodyEntry.tag_add("colored", "1.0", "end")

    # 배경색 서서히 변경
    update_background_color()


def send():
    try:
        username = temp_username.get()
        password = temp_password.get()
        to = temp_receiver.get()
        subject = temp_subject.get()
        body = temp_body.get()
        if username == "" or password == "" or to == "" or subject == "" or body == "":
            notif.config(text="All fields required", fg="red")
            return
        else:
            finalMessage = 'Subject: {}\n\n{}'.format(subject, body)
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(username, password)
            server.sendmail(username, to, finalMessage)
            notif.config(text="Email has been sent successfully", fg="green")
    except:
        notif.config(text="Error sending email", fg="red")


def reset():
    usernameEntry.delete(0, 'end')
    passwordEntry.delete(0, 'end')
    receiverEntry.delete(0, 'end')
    subjectEntry.delete(0, 'end')
    bodyEntry.delete(0, 'end')


def extract_previous_sentence(event):
    text = bodyEntry.get("1.0", "end-1c")
    sentences = re.split(r'[.,?!]|\n', text)
    print(sentences)
    predict(sentences)


# Labels
Label(master, text="Custom Email App", font=(
    'Calibri', 15)).grid(row=0, sticky=N)
Label(master, text="Please use the form below to send an email",
      font=('Calibri', 11)).grid(row=1, sticky=W, padx=5, pady=10)

Label(master, text="Email", font=('Calibri', 11)).grid(row=2, sticky=W, padx=5)
Label(master, text="Password", font=('Calibri', 11)).grid(
    row=3, sticky=W, padx=5)
Label(master, text="To", font=('Calibri', 11)).grid(row=4, sticky=W, padx=5)
Label(master, text="Subject", font=('Calibri', 11)).grid(
    row=5, sticky=W, padx=5)
Label(master, text="Body", font=('Calibri', 11)).grid(row=6, sticky=W, padx=5)
notif = Label(master, text="", font=('Calibri', 11), fg="red")
notif.grid(row=7, sticky=S)

# Storage
temp_username = StringVar()
temp_password = StringVar()
temp_receiver = StringVar()
temp_subject = StringVar()
temp_body = StringVar()
state_var = StringVar()

# Entries
usernameEntry = Entry(master, textvariable=temp_username)
usernameEntry.grid(row=2, column=0)
passwordEntry = Entry(master, show="*", textvariable=temp_password)
passwordEntry.grid(row=3, column=0)
receiverEntry = Entry(master, textvariable=temp_receiver)
receiverEntry.grid(row=4, column=0)
subjectEntry = Entry(master, textvariable=temp_subject)
subjectEntry.grid(row=5, column=0)

bodyEntry = Text(master, wrap=WORD, height=10, width=20)
bodyEntry.grid(row=6, column=0)
bodyEntry.bind("<KeyRelease>", extract_previous_sentence)


# Buttons
Button(master, text="Send", command=send).grid(
    row=7,   sticky=W,  pady=15, padx=5)
Button(master, text="Reset", command=reset).grid(
    row=7,  sticky=W,  padx=45, pady=40)

# Mainloop
master.mainloop()
