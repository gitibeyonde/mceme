from prediction.predict import prediction
    

new_predict = prediction(model_path='/home/admin/ai/model3.json',model_weights='/home/admin/ai/model3.h5'
                                      ,height=32,width=32)
#new_1 = glob.glob('/Users/techsid/Documents/anL/*.jpg')
# new_2 = glob.glob('/Users/techsid/Documents/aL/*.jpg')
# new_1.extend(new_2)
current = cv2.imread(fname)
            gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            res = cv2.resize(gray, dsize=(self.height,self.width), interpolation=cv2.INTER_CUBIC)
#new_predict.mixedprep(new_1)

result = new_predict.predict(image = current)
print result