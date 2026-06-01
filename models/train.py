import os

if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/breast_cancer_model.h5')