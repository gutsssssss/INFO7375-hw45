import class_file as cf
import numpy as np
import arguments as arg


# Define the training and testing dataset
trainingSet = cf.Dataset(arg.img_height, arg.img_width, arg.num_classes)
trainingSet.loadData(arg.address1)
testingSet = cf.Dataset(arg.img_height, arg.img_width, arg.num_classes)
testingSet.loadData(arg.address2)

# Initialize the neurons
model = cf.Model(arg.img_size, 100, 10)

# Train the model using gradient descent
training = cf.Training(model, trainingSet, arg.batch_size)
training.train(1)

# Save model and plot loss-curve
np.savez("model.npz", M=model)
cf.plot_loss_curve(training)

# Make predictions on the testing set
# test = cf.Test(model, testingSet)
