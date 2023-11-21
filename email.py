
from tkinter import *
import smtplib

# Main Screen Init
master = Tk()
master.title = 'Custom Email App'

# Functions


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

# Entries
usernameEntry = Entry(master, textvariable=temp_username)
usernameEntry.grid(row=2, column=0)
passwordEntry = Entry(master, show="*", textvariable=temp_password)
passwordEntry.grid(row=3, column=0)
receiverEntry = Entry(master, textvariable=temp_receiver)
receiverEntry.grid(row=4, column=0)
subjectEntry = Entry(master, textvariable=temp_subject)
subjectEntry.grid(row=5, column=0)
# bodyEntry = Entry(master, textvariable=temp_body)
bodyEntry = Text(master, wrap=WORD, height=10, width=20)
bodyEntry.grid(row=6, column=0)

# Buttons
Button(master, text="Send", command=send).grid(
    row=7,   sticky=W,  pady=15, padx=5)
Button(master, text="Reset", command=reset).grid(
    row=7,  sticky=W,  padx=45, pady=40)

# Mainloop
master.mainloop()


# import smtplib
# import ssl

# port = 465  # For SSL
# smtp_server = "smtp.gmail.com"
# sender_email = "hwangbird99@gmail.com"  # Enter your address
# receiver_email = "hwangbird99@gmail.com"  # Enter receiver address
# password = input("Type your password and press enter: ")
# message = """\
# Subject: Hi there

# This message is sent from Python."""

# context = ssl.create_default_context()
# with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
#     server.login(sender_email, password)
#     server.sendmail(sender_email, receiver_email, message)
