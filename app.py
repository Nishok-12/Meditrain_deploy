from flask import Flask, render_template, request, session, url_for, flash, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import pandas as pd
from fuzzywuzzy import process
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import os
from datetime import datetime
import random
from sklearn.preprocessing import LabelEncoder
import re

app = Flask(__name__)
app.secret_key = 'your_secret_key'

last_two_messages=[]
lastt_two_messages=[]

Selected_disease=""
print(Selected_disease)


# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///meditrain.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    country = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    contact = db.Column(db.String(15), nullable=False)
    password = db.Column(db.String(100), nullable=False)

class DocRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Text, nullable=False)
    time = db.Column(db.Text, nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    disease = db.Column(db.String(100), nullable=False)

class PatRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Text, nullable=False)
    time = db.Column(db.Text, nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    disease = db.Column(db.String(100), nullable=False)
    accuracy=db.Column(db.Integer, nullable=False)

# Routes
@app.route('/')
def home():
    session.clear()
    return render_template('home.html')

@app.route('/lpage')
def lpage():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        session['role']=request.form['role']
        
        user = User.query.filter_by(email=email, password=password).first()

        if user:
            session['pid'] = user.id
            session['name'] = user.name
            session["age"]=user.age
            session["gender"]=user.gender
            session["location"]=user.location
            session["country"]=user.country
            session["email"]=user.email
            session["contact"]=user.contact

            role=session.get('role')
            if role=='Patient':
                return redirect(url_for('index'))
            elif role=='Doctor':
                return redirect(url_for('pindex'))
        else:
            flash('Invalid email or password', 'danger')
            return redirect(url_for('lpage'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            user = User(
                name=request.form['name'],
                age=request.form['age'],
                gender=request.form['gender'],
                location=request.form['location'],
                country=request.form['country'],
                email=request.form['email'],
                contact=request.form['contact'],
                password=request.form['password']
            )
            db.session.add(user)
            db.session.commit()
            flash('Registration successful', 'success')
            return redirect(url_for('lpage'))
        except:
            flash('Error during registration', 'danger')
            return redirect(url_for('lpage'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route("/chat")

def index():
    session['var']=2
    user_name = session.get('name', 'Guest')
    return render_template('chat.html',user_name=user_name)


@app.route("/get", methods=["GET", "POST"])
def chat():
    global last_two_messages
    
    if 'var' not in session:
        session['var'] = 2

   

    if session['var'] == 1:
        session['var'] = 2  # Proceed to the next step

    else:

        msg = request.form["msg"]
        last_two_messages.append(msg)
        aabb=last_two_messages[-2:]
        combined_messages = ",".join(aabb)
        input = combined_messages
        print(input)
        return get_Chat1_response(input)
        


def get_Chat1_response(text):
    
    #Step 1 getting input or symptoms from user

    
    
    reprompt="By the way, do you have any other symptoms you'd like to mention? Anything else that's been bothering you lately?"
    
    user_input=text

    #step 2 user NER to extract the symptoms from user


    # Step 2.1: Load symptoms from CSV (make sure it only contains the desired symptoms)

    dataset_directory = os.path.join(os.getcwd(), 'static', 'dataset')  # Assuming this is relative to the current script
    csv_path = os.path.join(dataset_directory, 'unique_symptoms.csv')

    df = pd.read_csv(csv_path)  # Replace with the path to your CSV file
    symptoms_list = df['Symptom'].dropna().tolist()  # Extract symptoms into a list


    # Step 2.2: Function to generate n-grams (bigrams)
    def generate_ngrams(text, n=2):
        """Generate n-grams from text."""
        vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
        ngrams = vectorizer.fit_transform([text])
        ngrams_list = vectorizer.get_feature_names_out()
        return ngrams_list

    # Step 2.3: Function to extract symptoms using fuzzy matching and n-grams
    def extract_symptoms_from_input(user_input):
        """
        Function to extract symptoms from user input using fuzzy matching and n-grams.
        Returns a string of matched symptoms, separated by commas.
        """
        try:
            # Normalize input text
            user_input = user_input.lower().strip()

            # Generate bigrams from the user input
            user_input_bigrams = generate_ngrams(user_input)

            # List to store matched symptoms
            matched_symptoms = []

            for symptom in symptoms_list:
                # Check if the symptom directly matches any bigram
                if symptom.lower() in user_input:
                    matched_symptoms.append(symptom)
                else:
                    # Use fuzzy matching to allow for misspellings or slight variations
                    match = process.extractOne(symptom.lower(), user_input_bigrams)
                    if match and match[1] >= 80:  # Adjust threshold as needed
                        matched_symptoms.append(symptom)

            # Post-processing: Optional, for more fine-tuning, e.g., removing irrelevant matches.
       
            

            # Return the matched symptoms as a single string, joined by commas
            return ', '.join(matched_symptoms)

        except ValueError as e:
            # Handle the error gracefully and return a message or empty string
            return ""

    # Example usage:
    user_input = user_input
    extracted_symptoms = extract_symptoms_from_input(user_input)


    #Step 3 using disease prediction model (custom build with logistic Regression ) to diagonise the disease using symptoms

    # Load the saved Logistic Regression model
    model_directory = os.path.join(os.getcwd(), 'static', 'model')  # Assuming this is relative to the current script
    m1_path = os.path.join(model_directory, 'logistic_regression_model.pkl')
    v1_path = os.path.join(model_directory, 'tfidf_vectorizer.pkl') 
    e1_path = os.path.join(model_directory, 'label_encoder.pkl')

    loaded_model = joblib.load(m1_path)

    # Load the saved TF-IDF vectorizer
    loaded_vectorizer = joblib.load(v1_path)

    # Load the saved LabelEncoder (if not already loaded)
    label_encoder = joblib.load(e1_path)

    # Example of using the loaded model for prediction
    sample_input = extracted_symptoms
    sample_tfidf = loaded_vectorizer.transform([sample_input])  # Transform input using TF-IDF

    # Predict the disease label
    predicted_label = loaded_model.predict(sample_tfidf)

    # Decode the predicted label to get the actual disease name(s)
    predicted_disease = label_encoder.inverse_transform(predicted_label)

    # Output the predicted disease as a string
    predicted_disease_string = predicted_disease[0]  # Since there's only one prediction, we get the first one

   

    #Step 4 using treatment paln model (custom made with logistic Regression ) to suggest a treatmnent plan for the diagnized disease
    
    m2_path = os.path.join(model_directory, 'treatment_prediction_model.joblib')
    v2_path = os.path.join(model_directory, 'vectorizer.joblib')


    loaded_model = joblib.load(m2_path)
    loaded_vectorizer = joblib.load(v2_path)

    # Get the disease input as a string
    sample_disease = predicted_disease_string

    # Transform the input disease using the loaded vectorizer
    sample_disease_tfidf = loaded_vectorizer.transform([sample_disease])

    # Predict the treatment using the loaded model
    predicted_precaution = loaded_model.predict(sample_disease_tfidf)

    # Handle multi-output predictions for a single input (list of lists)
    predicted_precautions = predicted_precaution[0]


    # Output the result with each precaution on a new line
    i = 1
    precaution_list = []  # List to store the formatted precautions

    for precaution in predicted_precautions:
        precaution_list.append(f" {i}. {precaution.strip()}")
        i += 1

    # Combine all precautions into a single string with line breaks
    str7 = "\n".join(precaution_list)

    final_message = (
    f"Hello there, I’m really sorry to hear that you're feeling unwell. After reviewing your symptoms, it seems you may be dealing with '{predicted_disease_string}'.\n"
    "I understand this might be worrying, but please know that we can take steps together to help you feel better. 😊💪\n\n"
    "Please follow the following steps to feel better and work towards your recovery:\n"
    f"{str7}\n\n"  # Assuming str7 is already defined with precaution steps
    "Remember, recovery takes time, but you're not alone in this journey. Stay positive! 🌟\n\n"
    "If you continue to feel worse even after following the prescribed treatment, please don't hesitate to reach out to our medical expert or visit your nearest hospital. Your health is our top priority, and we want to ensure you receive the care you need. Take care, and feel free to contact us if you need further assistance. 🏥💬\n\n"
    )

    thank="Thank you for trusting us with your health. We are always here for you, anytime you need. Serving you is a privilege to us, and we are just a reach-out away. 🙏💙"
    
    sorry = "Sorry, but based on the information you've provided, I am unable to provide a diagnosis at this time. 😕\n" \
        "To ensure that I can give you a more accurate and helpful response, please create a new chat and provide additional details about your symptoms. 💬\n" \
        "Your health is important, and I want to make sure I can help as much as possible.💪\n  Don't hesitate to include any changes you've noticed in your condition or any other details that might be important. 🩺\n" \
        "Thank you for understanding, and I look forward to assisting you further once you've provided more symptoms! 🙏"
        
    if session['var'] == 2:
        session['var'] = 3
        return reprompt
    
    elif session['var'] == 3:
        session['var'] = 4
        if not extracted_symptoms.strip():
            print("Debug: No symptoms extracted. Returning 'sorry' message.")
            return sorry
        else:
            user = session.get('name', 'Guest')
            if user!='Guest':
                now = datetime.now()
                date = now.strftime("%d-%m-%Y")  # Custom date format
                time = now.strftime("%H:%M")

                sympt = extracted_symptoms.replace("_", " ")
                dis=predicted_disease_string
                p=session.get('pid')

                new_record = DocRecord(
                    user_id=p,
                    date=date,
                    time=time,
                    symptoms=sympt,
                    disease=dis
                )
                db.session.add(new_record)
                db.session.commit()

            return final_message
    
    elif session['var'] == 4:
        session['var'] = 5  # Or reset to 1 depending on your flow
        return thank

    
    else:
        session['var'] = 2 


@app.route("/pchat")

def pindex():
    global Selected_disease
    
    session['var1']=2
    Selected_disease=random_disease()
    user_name = session.get('name', 'Guest')
    return render_template('pchat.html',user_name=user_name)

def random_disease():
    dataset_directory = os.path.join(os.getcwd(), 'static', 'dataset')  # Assuming this is relative to the current script
    dcsv_path = os.path.join(dataset_directory, 'disease.csv')
    df = pd.read_csv(dcsv_path)
    disease_list = df['Disease'].dropna().tolist()
    Selected_disease = random.choice(disease_list)
    print(f"Selected Disease: {Selected_disease}")
    return Selected_disease


@app.route("/pget", methods=["GET", "POST"])
def pchat():
    global lastt_two_messages
    global Selected_disease
    
    if 'var1' not in session:
        session['var1'] = 2

    if session['var1']:

        msg = request.form["msg"]
        print(msg)
        lastt_two_messages.append(msg)
        aabb=lastt_two_messages[-2:]
        combined_messages = ",".join(aabb)
        input = combined_messages
        print(input)

        #Step 1 getting input or symptoms from user
        disease=Selected_disease
        return get_Chat2_response(input,disease)
    
def get_Chat2_response(text,disease):
    
    Selected_disease=disease

    dataset_directory = os.path.join(os.getcwd(), 'static', 'dataset')  # Assuming this is relative to the current script
    dcsv_path = os.path.join(dataset_directory, 'disease.csv')
    model_directory = os.path.join(os.getcwd(), 'static', 'model')  # Assuming this is relative to the current script
    m1_path = os.path.join(model_directory, 'logistic_regression222_model.pkl')
    v1_path = os.path.join(model_directory, 'tfidf222_vectorizer.pkl') 

    # Load the saved Logistic Regression model
    loaded_model = joblib.load(m1_path)

    # Load the saved TF-IDF vectorizer
    loaded_vectorizer = joblib.load(v1_path)

    # Load the CSV dataset for symptoms and diseases (same as training data)
    scsv_path = os.path.join(dataset_directory, 'symptoms.csv')
    df = pd.read_csv(scsv_path)

    # Check if 'Disease_Label' exists in the DataFrame
    if 'Disease_Label' not in df.columns:
        # If the 'Disease_Label' doesn't exist, encode the disease labels
        label_encoder = LabelEncoder()
        df['Disease_Label'] = label_encoder.fit_transform(df['Disease'])

    # Function to predict symptoms for a given disease
    def get_symptoms_for_disease(disease_name):
        # Encode disease name to its label (using LabelEncoder from the training process)
        disease_label = label_encoder.transform([disease_name])[0]  # Get the encoded label

        # Filter dataset to find symptoms for the given disease label
        disease_data = df[df['Disease_Label'] == disease_label]
        
        if disease_data.empty:
            return f"No data found for disease: {disease_name}"
        
        # Get symptoms associated with the given disease
        symptoms = disease_data['Symptoms'].values[0]
        symptom_1 = disease_data['symptom 1'].values[0]
        return symptoms, symptom_1

    # Example of how to use the function
    disease_name = Selected_disease  # User provides the disease name
    result, symptom_1 = get_symptoms_for_disease(disease_name)

    # Print the symptoms associated with the disease
    patient=result
    print(patient)
    s=session.get('var1')
    print(s)
    symot=symptom_1

    dis = Selected_disease  # Input disease name

    # Load the saved model and vectorizer# Assuming this is relative to the current script
    m2_path = os.path.join(model_directory, 'enhanced_treatment_prediction_model.joblib')
    v2_path = os.path.join(model_directory, 'enhanced_vectorizer.joblib') 
    loaded_model = joblib.load(m2_path)
    loaded_vectorizer = joblib.load(v2_path)

    # Transform the input disease using the loaded vectorizer
    sample_disease_tfidf = loaded_vectorizer.transform([dis])

    # Predict the treatment using the loaded model
    predicted_precautions = loaded_model.predict(sample_disease_tfidf)

    # Initialize a list to store the predicted precautions
    predicted_precaution_list = []

    # Extract precautions from the prediction
    for precaution_index, precaution in enumerate(predicted_precautions[0], start=1):
        # Ensure no leading/trailing spaces and handle cases of empty strings
        if precaution.strip():
            predicted_precaution_list.append(f"{precaution.strip()}")

    # Output the result as a list
    print(f"Predicted treatments for '{dis}':", predicted_precaution_list)

    user=text

    # Step 1: Load symptoms from CSV (make sure it only contains the desired symptoms)
    df = pd.read_csv(dcsv_path)  # Replace with the path to your CSV file
    disease_list = df['Disease'].dropna().tolist()  # Extract symptoms into a list

    ucsv_path = os.path.join(dataset_directory, 'unique_treatments.csv')
    dff = pd.read_csv(ucsv_path)  # Replace with the path to your CSV file
    treatment_list = dff['Treatment'].dropna().tolist()  # Extract symptoms into a list

    # Step 2: Function to generate n-grams (bigrams)
    def generate_ngrams(text, n=2):
        """Generate n-grams from text."""
        vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
        ngrams = vectorizer.fit_transform([text])
        ngrams_list = vectorizer.get_feature_names_out()
        return ngrams_list


    # Step 3: Function to extract symptoms using fuzzy matching and n-grams
    def extract_disease_from_input(user_input):
        
        # Normalize input text
        user_input = user_input.lower().strip()

        # Generate bigrams from the user input
        user_input_bigrams = generate_ngrams(user_input)

        # List to store matched disease
        matched_disease = []

        for disease in disease_list:
            # Check if the disease directly matches any bigram
            if disease.lower() in user_input:
                matched_disease.append(disease)
            else:
                # Use fuzzy matching to allow for misspellings or slight variations
                match = process.extractOne(disease.lower(), user_input_bigrams)
                if match and match[1] >= 95:  # Adjust threshold as needed
                    matched_disease.append(disease)

        # Post-processing: Optional, for more fine-tuning, e.g., removing irrelevant matches.
        matched_disease = list(set(matched_disease))  # Remove duplicates

        # Return the matched disease as a single string, joined by commas
        return ', '.join(matched_disease)

    def extract_treatment_from_input(user_input):

        # Normalize input text
        user_input = user_input.lower().strip()

        # Generate bigrams from the user input
        user_input_bigrams = generate_ngrams(user_input)

        # List to store matched treatment
        matched_treatments = []

        for treatment in treatment_list:
            # Check if the treatment directly matches any bigram
            if treatment.lower() in user_input:
                matched_treatments.append(treatment)
            else:
                # Use fuzzy matching to allow for misspellings or slight variations
                match = process.extractOne(treatment.lower(), user_input_bigrams)
                if match and match[1] >= 90:  # Adjust threshold as needed
                    matched_treatments.append(treatment)

        # Post-processing: Optional, for more fine-tuning, e.g., removing irrelevant matches.
        matched_treatments = list(set(matched_treatments))  # Remove duplicates

        # Return the matched symptoms as a single string, joined by commas
        return matched_treatments


    try:
        # Example usage:
        user_input = user
        extracted_disease = extract_disease_from_input(user_input)
        extracted_treatment = extract_treatment_from_input(user_input)

        print("Extracted Disease:", extracted_disease)
        print("Extracted Treatment:", extracted_treatment)

        accuracy_score = 0
        treatment_match_count = 0

        # Check if the extracted disease matches the selected disease
        if Selected_disease == extracted_disease:
            accuracy_score += 70  # Award 70 points for correct disease identification

            # Check for matching treatments
            for predicted_treatment in predicted_precaution_list:
                for extracted_treat in extracted_treatment:
                    if predicted_treatment == extracted_treat:
                        treatment_match_count += 1

            # Award 30 points if at least one treatment matches
            if treatment_match_count >= 1:
                accuracy_score += 30
        else:
            accuracy_score = 0  # No points if disease doesn't match

        # Output the results
        final=(f"Thank you Doctor for diagnosing and suggesting treatments for me. I am happy to share your accuracy.\nYour accuracy is about {accuracy_score}.")
        print(final)
    except Exception as e:
        # Handle exceptions and set accuracy score to 0
        print("An error occurred:", str(e))
        accuracy_score = 0
        final=(f"Thank you Doctor for diagnosing and suggesting treatments for me. I am happy to share your accuracy.\nYour accuracy is about {accuracy_score}.")
        print(final)

    if session['var1'] == 2:
        session['var1'] = 3
        print("returning patient")
        return patient
    
    elif session['var1'] == 3:
            session['var1'] = 4
        
            user = session.get('name', 'Guest')
            if user!='Guest':
                now = datetime.now()
                date = now.strftime("%d-%m-%Y")  # Custom date format
                time = now.strftime("%H:%M")

                sympt = symot
                dis=extracted_disease
                p=session.get('pid')
                ac=accuracy_score

                new_record = PatRecord(
                    user_id=p,
                    date=date,
                    time=time,
                    symptoms=sympt,
                    disease=dis,
                    accuracy=ac
                )

                db.session.add(new_record)
                db.session.commit()

            return final    

    
    else:
        session['var1'] = 2 


@app.route("/profile")
def profile():
    
        user_name = session.get('name', 'Guest')
        print(user_name)
        if user_name=='Guest':
            flash("You must login to access your Profile","danger")
            return redirect(url_for("lpage"))
        else:
            role=session.get('role')
            if role=='Patient':
                gen=session.get('gender')
                if gen=="Male":
                    ren="https://www.freeiconspng.com/uploads/male-icon-32.png"
                else:
                    ren="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRXSTblEVkkdJh15jlAbC3FpvuzCKb1o-pQQA&s"

                pdata = {
                "patient_id": session.get('pid'),
                "name": session.get('name'),
                "age": session.get('age'),
                "gender": session.get('gender'),
                "country": session.get('country'),
                "contact": session.get('contact'),
                "email": session.get('email'),
                "location": session.get('location'),
                "image":ren,
                "role":"Doctor",
                "url":url_for('index'),
                "roleas":role
                }

                headings=("Date","Time","Symptoms","Diagonised Disesase")
                
                patient_id = session.get('pid')
                # Query the records table using SQLAlchemy ORM
                records = DocRecord.query.filter_by(user_id=patient_id).all()
                rows = [(record.date, record.time, record.symptoms, record.disease) for record in records]
                data = tuple((row[0], row[1], row[2], row[3]) for row in rows)

                return render_template('profile.html',pdata=pdata,data=data,headings=headings)

            elif role=='Doctor':

                    gen=session.get('gender')
                    if gen=="Male":
                        ren="https://www.freeiconspng.com/uploads/male-icon-32.png"
                    else:
                        ren="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRXSTblEVkkdJh15jlAbC3FpvuzCKb1o-pQQA&s"

                    pdata = {
                    "patient_id": session.get('pid'),
                    "name": session.get('name'),
                    "age": session.get('age'),
                    "gender": session.get('gender'),
                    "country": session.get('country'),
                    "contact": session.get('contact'),
                    "email": session.get('email'),
                    "location": session.get('location'),
                    "image":ren,
                    "role":"Patient",
                    "url":url_for('pindex'),
                    "roleas":role
                    }

                    headings=("Date","Time","Symptoms","Diagonised Disesase","Accuracy Score")

                    patient_id = session.get('pid')
                    # Query the records table using SQLAlchemy ORM
                    records = PatRecord.query.filter_by(user_id=patient_id).all()
                    rows = [(record.date, record.time, record.symptoms, record.disease,record.accuracy) for record in records]
                    data = tuple((row[0], row[1], row[2], row[3],row[4]) for row in rows)

                    return render_template('profile.html',pdata=pdata,data=data,headings=headings)

if __name__ == '__main__':
    app.run(host"0.0.0.0",port=9999)
