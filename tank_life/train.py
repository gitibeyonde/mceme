import http.server
import socketserver
import urllib
import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import traceback


logging.getLogger('tensorflow').setLevel(logging.DEBUG)

ENGINE_LIFE=950
PORT_NUMBER=8080

df_train="./data.csv" #310 rows
df_eval="./eval.csv" #310 rows


COLUMNS=["Ser","EngHrs","Vibration","CoolantTemp","OilPressure"] #Ser,EngHrs,Vibration,CoolantTemp,OilPressure
RECORDS_ALL=[[0.0], [0.0], [0.0], [0.0],[0.0]]            

def input_fn(data_file, batch_size, num_epoch=None):                
    # Step 1                
    def parse_csv(value):        
        columns = tf.decode_csv(value, record_defaults=RECORDS_ALL)
        features = dict(zip(COLUMNS, columns))                
        features.pop('Ser')        
        labels =  features.pop('EngHrs') 
        return features, labels                            
          
    # Extract lines from input files using the Dataset API.    
    dataset = (tf.data.TextLineDataset(data_file) # Read text file       
          .skip(1) # Skip header row       
          .map(parse_csv)) 
    dataset = dataset.repeat(num_epoch)    
    dataset = dataset.batch(batch_size)                 
    # Step 3    
    iterator = dataset.make_one_shot_iterator()    
    features, labels = iterator.get_next() 
    print("Features %s, Labels %s"%(features, labels))   
    return features, labels    

next_batch = input_fn(df_train, batch_size=1, num_epoch=None)
with tf.Session() as sess:    
    first_batch  = sess.run(next_batch)    
    print(first_batch) 

#{'Vibration': <tf.Tensor 'IteratorGetNext:2' shape=(?,) dtype=float32>, 
#'CoolantTemp': <tf.Tensor 'IteratorGetNext:0' shape=(?,) dtype=float32>, 
#'OilPressure': <tf.Tensor 'IteratorGetNext:1' shape=(?,) dtype=float32>}, 
#Labels Tensor("IteratorGetNext:3", shape=(?,), dtype=float32, device=/device:CPU:0)
X1=tf.feature_column.numeric_column('Vibration')
X2=tf.feature_column.numeric_column('CoolantTemp')
X3=tf.feature_column.numeric_column('OilPressure')

base_columns = [X1, X2, X3]

model=tf.estimator.LinearRegressor(feature_columns=base_columns, model_dir='train3')  

# Train the estimator
model.train(steps=1000, input_fn=lambda: input_fn(df_train, batch_size=311, num_epoch=None))

results = model.evaluate(steps=None,input_fn=lambda: input_fn(df_eval, batch_size=27, num_epoch=1))

for key in results:   
    print("   {}, was: {}".format(key, results[key]))     
   
#"Vibration","CoolantTemp","OilPressure"
#328,272,75,9.5
#863,325,65,8.2 
prediction_input = {                
          'Vibration': [325, 272],                
          'CoolantTemp': [65,75],                
          'OilPressure': [8.2,9.2]
     }

print(prediction_input)
   
class Handler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        self.connection.settimeout(1)
        if self.path.endswith('/'):
            self.send_response(200)
            self.send_header('Content-type','text/html')
            self.end_headers()
            self.wfile.write(bytes('<html><head></head><body><h2>Predictive Maintenance of Tank Engine</h2><form action ="/" method="get"> \
                              Vibration: <input type="text" name="vibration"><br> \
                              Coolent Temp: <input type="text" name="coolent_temp"><br> \
                              Oil Pressure: <input type="text" name="oil_pressure"><br> \
                              <input type="submit" value="Submit"> \
                              </form></body></html>', "utf8"))
        else:
            o = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(o.query)
            prediction_input = {                
                'Vibration': [float(q['vibration'][0])],
                'CoolantTemp': [float(q['coolent_temp'][0])],
                'OilPressure': [float(q['oil_pressure'][0])],
            }
            print(prediction_input)
            
            def test_input_fn():    
                dataset = tf.data.Dataset.from_tensors(prediction_input)    
                return dataset
     
            pred_results = model.predict(input_fn=test_input_fn)

            for pred in enumerate(pred_results):    
                print(pred)  
                print(pred[1]['predictions'])  
                print("Engine remaining life is %d"%(900-pred[1]['predictions']))

        return

try:
    print('Server listening on port 8000...')
    httpd = socketserver.TCPServer(('', 8000), Handler)
    httpd.serve_forever()
except KeyboardInterrupt:
        httpd.socket.close()
