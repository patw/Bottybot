# This is used for hosting on a WSGI compliant server
# gUnicorn seems to work fine

from app import app

if __name__ == '__main__':
    app.run(debug=False)
