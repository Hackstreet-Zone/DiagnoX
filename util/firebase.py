import pyrebase

firebaseConfig = {
  apiKey: "AIzaSyB-Zcd-Uo6ABL5C2kiyGp6PWv9Df2vx4u0",
  authDomain: "diagnox-7d71a.firebaseapp.com",
  projectId: "diagnox-7d71a",
  storageBucket: "diagnox-7d71a.appspot.com",
  messagingSenderId: "773020495123",
  appId: "1:773020495123:web:da10c3ba9960484764fee6",
  measurementId: "G-Z4D124N7HN"
}

firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()
image = "uploads/00000020_001.png"


if __name__ == "__main__":
    storage.child(image).put(image)
# Get url of the uploaded image
    url = storage.child(image).get_url()