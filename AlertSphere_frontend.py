# Importing Model
import joblib
mnb= joblib.load('alertsphere_model.pkl')
cv = joblib.load('count_vectorizer.pkl')


import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

root = tk.Tk()
root.title('AlertSphere')

root.geometry("850x270")

# Set the icon
root.iconbitmap("images/icon.ico")

#for logo
logo_open = Image.open("images/logo.png")
size = (150, 150)
logo_resize = logo_open.resize(size)
logo = ImageTk.PhotoImage(logo_resize)

logo_label = tk.Label(root, image=logo)
logo_label.pack(side="top")

# for logo heading
heading = tk.Label(root, text="Disaster Tweet Detector", font=("Times New Roman", 24, 'bold'), fg='#219ebc', bg='#ffb703')
heading.pack(side="top")

# to hold all widgets
frame = tk.Frame(root)
frame.place(relx=0.5, rely=0.6, anchor='center')

# listbox for recent tweets
recent_tweets_frame = ttk.Frame(frame)
recent_tweets_frame.grid(row=1, column=2, rowspan=4, padx=10)

recent_tweets_scrollbar_y = tk.Scrollbar(recent_tweets_frame, orient=tk.VERTICAL)
recent_tweets_scrollbar_x = tk.Scrollbar(recent_tweets_frame, orient=tk.HORIZONTAL)
recent_tweets = tk.Listbox(recent_tweets_frame, width=60, height=10, yscrollcommand=recent_tweets_scrollbar_y.set, xscrollcommand=recent_tweets_scrollbar_x.set)
recent_tweets_scrollbar_y.config(command=recent_tweets.yview)
recent_tweets_scrollbar_x.config(command=recent_tweets.xview)

recent_tweets.grid(row=0, column=0, sticky="nsew")
recent_tweets_scrollbar_y.grid(row=0, column=1, sticky="ns")
recent_tweets_scrollbar_x.grid(row=1, column=0, sticky="ew")


# "Recent Tweets" heading inside the listbox 
recent_tweets.insert(0, "Recent Tweets:")
recent_tweets.itemconfig(0, {})
recent_tweets.config(state=tk.DISABLED)

# for credit
def show_credit_window():
    credit_window = tk.Toplevel(root)
    credit_window.title("Credit")
    credit_window.geometry("300x100")
    credit_label = tk.Label(credit_window, text= "Al Saim Shakeel (Project Leader) \n Mohd. Faraz Khan \n Ahtisham Riyasat")
    credit_label.pack(pady=20)   
    credit_window.iconbitmap("images/icon.ico") 

def predict_result(event=None):
    tweet_text = tweet_entry.get().strip()
    if not tweet_text:
        messagebox.showwarning("Alert", "Please enter the tweet.")
        return

    new_data = [tweet_text]

    # Update recent tweet listbox
    recent_tweets.config(state=tk.NORMAL)
    recent_tweets.insert(tk.END, new_data[0])

    # cv and mnb are defined above in your code

    new_data_cv = cv.transform(new_data)

    pred_mnb = mnb.predict(new_data_cv)

    if pred_mnb == 1:
        result = 'Disaster Tweet'
        recent_tweets.itemconfig(tk.END, {'fg': 'red'})
        messagebox.showinfo("Prediction Result", result, icon='error')
    elif pred_mnb == 0:
        result = 'Normal Tweet'
        recent_tweets.itemconfig(tk.END, {'fg': 'green'})
        messagebox.showinfo("Prediction Result", result, icon='info')

    recent_tweets.config(state=tk.DISABLED)
    tweet_entry.delete(0, 'end')
    
    
tk.Label(frame, text="Tweet:", font=("Arial", 14),bg='#d3e0ea').grid(row=0, column=0)
tweet_entry = tk.Entry(frame, width=50)
tweet_entry.grid(row=0, column=1)

submit_button = tk.Button(frame, text="Detect", command=predict_result, font=("Arial", 14), bg="#219ebc", fg="white")
submit_button.grid(row=2, columnspan=2)


# Button to show credit
credit_button = tk.Button(root, text="Credits", command=show_credit_window, font=("Arial", 12), bg="#333333", fg="white")
credit_button.pack(anchor='se', side='bottom', pady=10)

# Return key ko attach kiya h
root.bind('<Return>', predict_result)

frame.config(borderwidth=2, relief="groove", bg="#d3e0ea", padx=20, pady=20)
tweet_entry.config(bg="#ffffff", highlightthickness=1, highlightbackground="#cccccc")
submit_button.config(bg="#219ebc", fg="#ffffff", padx=10, pady=5)
credit_button.config(bg="#333333", fg="#ffffff", padx=10, pady=5)
recent_tweets_frame.config(borderwidth=2, relief="groove")
recent_tweets.config(bg="#f0f0f0", highlightthickness=0)
root.mainloop()