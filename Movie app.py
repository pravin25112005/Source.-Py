#  AI Movie Matchmaking Streamlit App

import streamlit as st
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime

# Constants
DATA_URL = "https://raw.githubusercontent.com/amankharwal/Website-data/master/imdb_top_1000.csv"
POSTER_PLACEHOLDER = "https://via.placeholder.com/150x225?text=No+Poster"
MIN_PASSWORD_LENGTH = 6
XP_PER_RATING = 15
XP_PER_WATCH = 10
XP_LEVEL_THRESHOLD = 100

# Initialize users.json if it doesn't exist
if not os.path.exists("users.json"):
    with open("users.json", "w") as f:
        json.dump({}, f)

@st.cache_data
def load_movies():
    """Load movie data with enhanced error handling and data cleaning"""
    try:
        df = pd.read_csv(DATA_URL)
        # Data cleaning
        df['Runtime'] = df['Runtime'].str.extract('(\d+)').astype(float)
        df['Gross'] = pd.to_numeric(df['Gross'].str.replace('[^\d]', '', regex=True), errors='coerce')
        df['combined_features'] = df['Genre'].fillna('') + " " + df['Director'].fillna('') + " " + df['Overview'].fillna('')
        return df
    except Exception as e:
        st.error(f"Error loading movie data: {str(e)}")
        return pd.DataFrame({
            "Series_Title": ["Sample Movie 1", "Sample Movie 2"],
            "Genre": ["Action", "Comedy"],
            "Director": ["Director A", "Director B"],
            "Overview": ["Sample overview 1", "Sample overview 2"],
            "IMDB_Rating": [7.5, 8.0],
            "Poster_Link": [POSTER_PLACEHOLDER, POSTER_PLACEHOLDER],
            "Runtime": [120, 90],
            "combined_features": ["Action Director A Sample overview 1", "Comedy Director B Sample overview 2"]
        })

@st.cache_resource
def vectorize(df):
    """Create TF-IDF vectors for movie content"""
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = tfidf.fit_transform(df["combined_features"])
    return tfidf, matrix

def load_users():
    """Load user data with error handling"""
    try:
        with open("users.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading user data: {e}")
        return {}

def save_users(data):
    """Save user data with error handling"""
    try:
        with open("users.json", "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        st.error(f"Error saving user data: {e}")

def load_poster(url):
    """Load movie poster with error handling"""
    try:
        if url == POSTER_PLACEHOLDER:
            return Image.new('RGB', (150, 225), color='gray')
        response = requests.get(url, timeout=5)
        return Image.open(BytesIO(response.content))
    except:
        return Image.new('RGB', (150, 225), color='gray')

def register_user(username, password, users):
    """Register new user with validation"""
    if len(username) < 3:
        return "Username must be at least 3 characters"
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
    if username in users:
        return "Username already exists"
    
    users[username] = {
        "password": password,
        "friends": [],
        "watched": [],
        "continue_watching": [],
        "favorites": [],
        "xp": 0,
        "level": 1,
        "ratings": {},
        "chats": {},
        "join_date": datetime.now().strftime("%Y-%m-%d"),
        "preferences": {
            "genres": [],
            "actors": [],
            "directors": []
        }
    }
    save_users(users)
    return None

def get_movie_card(movie):
    """Create a styled movie card"""
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(load_poster(movie['Poster_Link']), width=150)
        with col2:
            st.subheader(movie['Series_Title'])
            st.caption(f"⭐ {movie['IMDB_Rating']} | {movie['Runtime']} min | {movie['Genre']}")
            st.write(movie['Overview'][:150] + "...")
    return st

def recommend_movies(user_data, df, matrix, tfidf, n=5):
    """Generate personalized recommendations with error handling"""
    try:
        if not user_data["watched"]:
            return df.sample(min(n, len(df)))
        
        watched_indices = df[df["Series_Title"].isin(user_data["watched"])].index
        if len(watched_indices) == 0:
            return df.sample(min(n, len(df)))
        
        watched_vec = matrix[watched_indices].mean(axis=0)
        sim_scores = cosine_similarity(watched_vec, matrix).flatten()
        top_indices = sim_scores.argsort()[-n-1:-1][::-1]
        return df.iloc[top_indices]
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return df.sample(min(n, len(df)))

# Initialize session state
if "user" not in st.session_state:
    st.session_state.user = None
if "page" not in st.session_state:
    st.session_state.page = "Login"
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "current_movie" not in st.session_state:
    st.session_state.current_movie = None

# Load data
df = load_movies()
tfidf, matrix = vectorize(df)
users = load_users()

# Page functions
def login_page():
    st.title("🎬 AI Movie Matchmaker")
    st.write("Find your perfect movie matches and connect with friends!")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if username in users and users[username]["password"] == password:
                st.session_state.user = username
                st.session_state.page = "Home"
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
    
    with tab2:
        new_user = st.text_input("Choose a username", key="reg_user")
        new_pass = st.text_input("Choose a password", type="password", key="reg_pass")
        confirm_pass = st.text_input("Confirm password", type="password", key="confirm_pass")
        
        if st.button("Create Account"):
            if new_pass != confirm_pass:
                st.error("Passwords don't match!")
            else:
                error = register_user(new_user, new_pass, users)
                if error:
                    st.error(error)
                else:
                    st.success("Account created! Please login.")

def home_page():
    st.sidebar.title(f"Welcome, {st.session_state.user}")
    user_data = users[st.session_state.user]
    
    # Navigation
    pages = {
        "🏠 Home": "Home",
        "🎥 Discover": "Discover",
        "👥 Friends": "Friends",
        "👤 Profile": "Profile"
    }
    selection = st.sidebar.radio("Menu", list(pages.keys()))
    st.session_state.page = pages[selection]
    
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.session_state.page = "Login"
        st.rerun()
    
    # Main content
    st.title("🎬 Your Movie Dashboard")
    
    # Continue Watching
    if user_data["continue_watching"]:
        st.subheader("Continue Watching")
        cols = st.columns(min(4, len(user_data["continue_watching"])))
        for idx, title in enumerate(user_data["continue_watching"][:4]):
            movie = df[df["Series_Title"] == title].iloc[0]
            with cols[idx % 4]:
                get_movie_card(movie)
                if st.button(f"Continue {title[:15]}...", key=f"cont_{idx}"):
                    st.session_state.current_movie = movie.to_dict()
                    st.session_state.page = "Watch"
                    st.rerun()
    
    # Recommendations
    st.subheader("Recommended For You")
    recs = recommend_movies(user_data, df, matrix, tfidf, 4)
    if len(recs) > 0:
        cols = st.columns(min(4, len(recs)))
        for idx, (_, movie) in enumerate(recs.iterrows()):
            with cols[idx % 4]:
                get_movie_card(movie)
                if st.button(f"Watch {movie['Series_Title'][:15]}...", key=f"rec_{idx}"):
                    users[st.session_state.user]["continue_watching"].append(movie["Series_Title"])
                    save_users(users)
                    st.session_state.current_movie = movie.to_dict()
                    st.session_state.page = "Watch"
                    st.rerun()
    else:
        st.warning("No recommendations available. Watch some movies first!")

def discover_page():
    st.title("🔍 Discover Movies")
    
    # Search and filters
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.search_query = st.text_input("Search movies", st.session_state.search_query)
    with col2:
        genre_filter = st.selectbox("Filter by genre", ["All"] + list(df['Genre'].str.split(',').explode().str.strip().unique()))
    
    # Display results
    filtered = df.copy()
    if st.session_state.search_query:
        mask = (
            filtered['Series_Title'].str.contains(st.session_state.search_query, case=False) |
            filtered['Director'].str.contains(st.session_state.search_query, case=False) |
            filtered['Overview'].str.contains(st.session_state.search_query, case=False)
        )
        filtered = filtered[mask]
    
    if genre_filter != "All":
        filtered = filtered[filtered['Genre'].str.contains(genre_filter, case=False)]
    
    st.subheader(f"Found {len(filtered)} movies")
    for _, movie in filtered.iterrows():
        get_movie_card(movie)
        if st.button(f"Watch {movie['Series_Title']}", key=f"dis_{movie['Series_Title']}"):
            users[st.session_state.user]["continue_watching"].append(movie["Series_Title"])
            save_users(users)
            st.session_state.current_movie = movie.to_dict()
            st.session_state.page = "Watch"
            st.rerun()

def watch_page():
    if st.session_state.current_movie is None:
        st.warning("No movie selected")
        st.session_state.page = "Home"
        st.rerun()
    
    movie = st.session_state.current_movie
    st.title(f"🎥 Watching: {movie['Series_Title']}")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(load_poster(movie['Poster_Link']), width=300)
    with col2:
        st.subheader(movie['Series_Title'])
        st.write(f"**Director:** {movie['Director']}")
        st.write(f"**Genre:** {movie['Genre']}")
        st.write(f"**Rating:** ⭐ {movie['IMDB_Rating']}")
        st.write(f"**Runtime:** {movie['Runtime']} minutes")
        st.write("**Overview:**")
        st.write(movie['Overview'])
    
    # Video player placeholder
    st.video("https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4")
    
    # Rating and actions
    with st.expander("Rate this movie"):
        rating = st.slider("Your rating", 1, 10, 5)
        if st.button("Submit Rating"):
            user = st.session_state.user
            users[user]["ratings"][movie['Series_Title']] = rating
            if movie['Series_Title'] not in users[user]["watched"]:
                users[user]["watched"].append(movie['Series_Title'])
                users[user]["xp"] += XP_PER_WATCH
            users[user]["xp"] += XP_PER_RATING
            save_users(users)
            st.success("Rating submitted!")
    
    if st.button("Back to Home"):
        st.session_state.page = "Home"
        st.rerun()

def friends_page():
    st.title("👥 Friends")
    username = st.session_state.user
    user_data = users[username]
    
    tab1, tab2 = st.tabs(["Your Friends", "Find Friends"])
    
    with tab1:
        st.subheader("Your Friends")
        if not user_data["friends"]:
            st.info("You don't have any friends yet. Add some below!")
        
        for friend in user_data["friends"]:
            with st.expander(friend):
                st.write(f"Member since: {users[friend].get('join_date', 'Unknown')}")
                st.write(f"Level: {users[friend].get('level', 1)}")
                
                # Display friend's recently watched
                st.write("Recently watched:")
                for movie in users[friend]["watched"][-3:]:
                    st.write(f"- {movie}")
                
                # Chat button
                if st.button(f"Chat with {friend}"):
                    st.session_state.chat_friend = friend
    
    with tab2:
        st.subheader("Find New Friends")
        search_term = st.text_input("Search by username")
        
        if search_term:
            results = [u for u in users if search_term.lower() in u.lower() and u != username]
            if not results:
                st.info("No users found")
            
            for user in results:
                with st.container():
                    st.write(f"**{user}** (Level {users[user].get('level', 1)})")
                    if user not in user_data["friends"]:
                        if st.button(f"Add {user}", key=f"add_{user}"):
                            user_data["friends"].append(user)
                            users[user]["friends"].append(username)
                            save_users(users)
                            st.success(f"Added {user} as a friend!")
                            st.rerun()
    
    # Chat interface
    if "chat_friend" in st.session_state:
        friend = st.session_state.chat_friend
        st.subheader(f"💬 Chat with {friend}")
        
        # Get chat history
        chat_key = "-".join(sorted([username, friend]))
        messages = user_data["chats"].get(chat_key, [])
        
        # Display messages
        for msg in messages[-10:]:  # Show last 10 messages
            align = "right" if msg["sender"] == username else "left"
            st.chat_message(align).write(f"{msg['sender']}: {msg['text']}")
        
        # Send new message
        new_msg = st.chat_input("Type your message")
        if new_msg:
            new_entry = {"sender": username, "text": new_msg, "time": datetime.now().strftime("%H:%M")}
            messages.append(new_entry)
            user_data["chats"][chat_key] = messages
            users[friend]["chats"][chat_key] = messages
            save_users(users)
            st.rerun()

def profile_page():
    st.title("👤 Your Profile")
    username = st.session_state.user
    user_data = users[username]
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Level", user_data["level"])
        st.progress(min(user_data["xp"] / XP_LEVEL_THRESHOLD, 1.0))
        st.caption(f"{user_data['xp']}/{XP_LEVEL_THRESHOLD} XP to next level")
        st.write(f"Member since: {user_data.get('join_date', 'Unknown')}")
    with col2:
        st.subheader("Your Stats")
        st.write(f"🎥 Movies watched: {len(user_data['watched'])}")
        st.write(f"⭐ Movies rated: {len(user_data['ratings'])}")
        st.write(f"❤️ Favorites: {len(user_data['favorites'])}")
        st.write(f"👥 Friends: {len(user_data['friends'])}")
    
    tab1, tab2, tab3 = st.tabs(["Your Movies", "Preferences", "Account"])
    
    with tab1:
        st.subheader("Your Watched Movies")
        for movie in user_data["watched"][-10:]:  # Show last 10 watched
            rating = user_data["ratings"].get(movie, "Not rated")
            st.write(f"- {movie} ({'⭐' * int(rating) if isinstance(rating, int) else rating})")
        
        st.subheader("Your Favorites")
        for movie in user_data["favorites"]:
            st.write(f"- {movie}")
    
    with tab2:
        st.subheader("Update Preferences")
        
        # Genre preferences
        selected_genres = st.multiselect(
            "Favorite genres",
            options=df['Genre'].str.split(',').explode().str.strip().unique(),
            default=user_data["preferences"]["genres"]
        )
        
        # Director preferences
        selected_directors = st.multiselect(
            "Favorite directors",
            options=df['Director'].unique(),
            default=user_data["preferences"]["directors"]
        )
        
        if st.button("Save Preferences"):
            user_data["preferences"]["genres"] = selected_genres
            user_data["preferences"]["directors"] = selected_directors
            save_users(users)
            st.success("Preferences updated!")
    
    with tab3:
        st.subheader("Account Settings")
        st.warning("Coming soon: Password change and account deletion")

# Main app router
def main():
    if st.session_state.user is None:
        login_page()
    else:
        if st.session_state.page == "Home":
            home_page()
        elif st.session_state.page == "Discover":
            discover_page()
        elif st.session_state.page == "Watch":
            watch_page()
        elif st.session_state.page == "Friends":
            friends_page()
        elif st.session_state.page == "Profile":
            profile_page()

if __name__ == "__main__":
    main()
