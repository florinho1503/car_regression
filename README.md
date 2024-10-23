In this project, I use a machine learning regression model to predict prices of second hand cars listed on 
autoscout24.nl to find undervalued cars. I first scrape the price and features of the different listings using Beautiful Soup.
I then store the cars along with its price and features in a database and train a random forest regressor model from
sklearn on the data to learn to predict the price Y from the other features X (mileage, car age, type etc.).
I use k-fold validation to make sure that we can use all cars not only for training but also for testing, 
because the goal is to find undervalued cars. During testing, the model predicts the car prices, and the most
undervalued ones according to the model are printed out along with the link on where to find the car.