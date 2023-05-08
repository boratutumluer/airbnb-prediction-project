from airbnb import app

# Checks if the run.py file has executed directly and not imported
if __name__ == '__main__':
    with app.app_context():
        app.run(debug=True)
