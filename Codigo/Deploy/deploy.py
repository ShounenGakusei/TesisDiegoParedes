from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from createModel import crearModelo


app = Flask(__name__)
api = Api(app)


model = crearModelo()
model.load_weights('Model_20220417_081105.hdf5')

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')



class PredictSentiment(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        """
        Conseguimos las imagenes 
        """     
        ###   TODO
        prediction = model.predict(user_query)  
        # Output 'Negative' or 'Positive' along with the score
        if prediction == 0:
            pred_text = 'Negative'
        else:
            pred_text = 'Positive'
            
        # round the predict proba value and set to new variable
        confidence = round(pred_proba[0], 3)
        
        # create JSON object
        output = {'prediction': pred_text, 'confidence': confidence}
        
        return output

if __name__ == '__main__':
    app.run(debug=True)