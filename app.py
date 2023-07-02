from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample data (in-memory)
books = [
    {"id": 1, "title": "Book 1", "author": "Author 1"},
    {"id": 2, "title": "Book 2", "author": "Author 2"},
    {"id": 3, "title": "Book 3", "author": "Author 3"}
]

# GET /books - Get all books
@app.route('/books', methods=['GET'])
def get_books():
    return jsonify(books)
