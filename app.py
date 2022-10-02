# 1. Library imports 
import uvicorn   # pip install  uvicorn
from fastapi import FastAPI    # pip install fastapi
from BankNotes import BankNote
import pickle    # pip install pickle
# 2. Create app object and read pickle
app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)
# 3. index route
@app.get('/')
def index():
    return {'message': '기계학습 API - 위조지폐 인식'}
# 4. predict route
@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    if(prediction[0]>0.5):
        prediction="not fake"
    else:
        prediction="fake"
    return { 'prediction': prediction }
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn app:app --reload