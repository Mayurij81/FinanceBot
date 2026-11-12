FinanceGURU 💰

An AI chatbot for personalised Indian financial planning. It asks about your age and goals, then suggests an investment mix across equity, gold and savings, in simple language and in ₹.

🏆 Top 50, Mercer Mettl Hackathon 2025 <!-- add "out of N teams" if you know -->


![WhatsApp Image 2025-06-18 at 13 53 33_2ebb171e](https://github.com/user-attachments/assets/6c54358b-514e-486a-ba20-bf74f351c7d1)


![image](https://github.com/user-attachments/assets/1100c35e-9fd0-4a75-84f9-7046dc4b2608)


## Features
- Age- and goal-based portfolio planning (equity, gold and savings mix)
- Indian financial context: ₹, gold, property
- Retrieval-augmented advice: relevant financial knowledge is fetched from ChromaDB and passed to Mistral, so answers are grounded instead of generic
- Input validation and adaptive dialogue, so the bot keeps the conversation on track and asks follow-up questions when details are missing
- Plain-language explanations for beginners
## How It Works
1. The user chats with the React frontend.
2. The Flask API validates the input and builds an embedding with SentenceTransformers.
3. ChromaDB returns the most relevant financial knowledge.
4. That context plus the user's profile goes to the Mistral API, which writes the advice.
## Tech Stack
- **Backend:** Python, Flask
- **AI:** Mistral AI API, SentenceTransformers, ChromaDB (vector search)
- **Frontend:** React
## Setup
 
### 1. Clone the repo
```bash
git clone https://github.com/Mayurij81/FinanceBot.git
cd FinanceBot
```
 
### 2. Backend
```bash
python -m venv myenv
source myenv/bin/activate        # Windows: myenv\Scripts\activate
pip install -r requirements.txt
```
 
Create a `.env` file (never commit it) with your key:
```
MISTRAL_API_KEY=your_key_here
```
 
Run the server:
```bash
python app.py   # adjust to your entry file name
```
 
### 3. Frontend
```bash
cd frontend     # adjust to your folder name
npm install
npm start
```
 
## Disclaimer
FinanceGURU is a student project for learning purposes and does not provide professional financial advice.
 
## Author
Mayuri Jadhav · [LinkedIn](#) · [GitHub](https://github.com/Mayurij81)
