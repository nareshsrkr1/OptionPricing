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

# GET /books/<book_id> - Get a specific book
@app.route('/books/<int:book_id>', methods=['GET'])
def get_book(book_id):
    book = next((book for book in books if book['id'] == book_id), None)
    if book:
        return jsonify(book)
    else:
        return jsonify({'error': 'Book not found'}), 404

# POST /books - Create a new book
@app.route('/books', methods=['POST'])
def create_book():
    new_book = {
        'id': len(books) + 1,
        'title': request.json['title'],
        'author': request.json['author']
    }
    books.append(new_book)
    return jsonify(new_book), 201

# PUT /books/<book_id> - Update a book
@app.route('/books/<int:book_id>', methods=['PUT'])
def update_book(book_id):
    book = next((book for book in books if book['id'] == book_id), None)
    if book:
        book['title'] = request.json.get('title', book['title'])
        book['author'] = request.json.get('author', book['author'])
        return jsonify(book)
    else:
        return jsonify({'error': 'Book not found'}), 404

# DELETE /books/<book_id> - Delete a book
@app.route('/books/<int:book_id>', methods=['DELETE'])
def delete_book(book_id):
    book = next((book for book in books if book['id'] == book_id), None)
    if book:
        books.remove(book)
        return jsonify({'message': 'Book deleted'})
    else:
        return jsonify({'error': 'Book not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0')
