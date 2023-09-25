# Immo Eliza Project: Property Value Estimation

## Description

This project aims to create an advanced AI model capable of estimating the value of houses and apartments for sale in Belgium based on information provided by users. With this AI-powered solution, potential buyers, sellers, and real estate professionals can access property worth predictions to make better decisions on the dynamic currently real estate market.

The development process is thoughtfully divided into four parts, each addressing crucial aspects of the project:

**Part 1:** *Data Acquisition*

In the first part of the project, it was performed a web scraping to acquire a dataset with the data of approximately 10,000 properties. After this, the data is saved in a CSV file named dataset-immo.csv and stored in the folder data.

**Part 2:** *Data Cleaning, Analysis, and Interpretation*

The second part of the project involves data cleaning, analysis, and interpretation. The acquired dataset is processed to handle missing values, outliers, inconsistencies, and irrelevant information to ensures that the dataset is in a suitable state for further analysis. Next, exploratory data analysis (EDA) is carried out to gain insights into the dataset's characteristics. Through statistical calculations, visualizations, and data summarization, important features are identified, relationships between variables, and potential predictors of property value. The insights obtained during this phase guide the development of the AI model.

**Part 3:** *Machine Learning with Random Forest Regressor*

![alt text](outputs/random_fores_top_20_update.png)

> In this picture it is possible to see the top 20 features that impact the most the acquired model.

In the third part of the project, a machine learning model using the Random Forest Regressor is created. This ensemble learning algorithm is well-suited for regression tasks and can handle numerical and categorical features. The relevant features based on the insights gained from Part 2 were selected and the dataset was further splitted into training and testing sets. The training set is used to train the Random Forest Regressor to learn the relationships between features and property values. After training, the model's accuracy was evaluated using Mean Squared Error (MSE) and R-squared (R2) score. The R2 score indicates how well the model fits the data, providing valuable information about its performance.

**Part 4:** *Model Deployment and Integration*

In the final part, it was deployed the AI model as an API using FastAPI and it was containerized with Docker for portability. The AI model is accessible through a public URL with personalized port on Render. Users can make property value estimations by sending data to the API and receiving the results in real-time. This integration and deployment process make the property value estimation AI model a practical and user-friendly tool for potential property buyers, sellers, and real estate professionals in Belgium.

## Step-by-Step Instructions with API and Docker

1. Clone the Repository: first, clone the project repository to your local machine.
2. Build a Docker Image: open a terminal or command prompt and navigate to the directory containing the Dockerfile. Run the following command to build the Docker image: docker build . -t api_predict
3. Run Docker Container: after building the Docker image, execute the following command to run the Docker container: docker run -p 8000:8000 api_predict
4. Access API Documentation: On the terminal the following line will be displayed: INFO: Uvicorn running on http://0.0.0.0:8000
Open a web browser and paste the provided link http://0.0.0.0:8000 into the address bar.
5. Explore API Endpoints: Through this link the documentation for the API is accessed where all available routes and endpoints are outlined.
6. Access Interactive Swagger Page: Update the link to http://0.0.0.0:8000/docs in the address bar of your web browser.
7. Perform Property Value Estimation: On the Swagger page, locate the POST route for the prediction endpoint. Click on "Try it out."
8. Modify Input Data: Modify the JSON input dictionary with property details accordingly. Replace the example data with specific property information.
9. Execute Prediction: Click on "Execute" to submit the modified JSON input and receive the property value prediction.

## Step-by-Step Instructions Accessing the Property Value Estimation App

1. Access the Deployed App: No need to create an image or container! Click on the following link to directly access the deployed Property Value Estimation App: https://immo-prediction-api.onrender.com/
2. Explore API Endpoints: Upon accessing the link, the documentation for the API can be found. This documentation outlines all available routes and endpoints.
3. Access Interactive Swagger Page: Update the link to https://immo-prediction-api.onrender.com/docs in the address bar of your web browser.
4. Perform Property Value Estimation: On the Swagger page, locate the POST route for the prediction endpoint. Click on "Try it out."
5. Modify Input Data: Modify the JSON input dictionary with property details accordingly. Replace the example data with specific property information.
6. Execute Prediction: Click on "Execute" to submit the modified JSON input and receive the property value prediction.

## Main Technologies Used

**Part 1:** *Data Acquisition*

* Beautiful Soup (Latest version: 4.9.3)
* Requests (Latest version: 2.26.0)
* Concurrency (Latest version: 1.2.1)
* Pandas (Latest version: 1.3.3)

**Part 2:** *Data Cleaning, Analysis and Interpretation*

* Seaborn (Latest version: 0.11.2)
* Matplotlib (Latest version: 3.4.3)
* NumPy (Latest version: 1.21.3)
* OS (Latest version: 3.11.2)

**Part 3:** *Machine Learning with Random Forest Regressor*

* Scikit-learn (Latest version: 0.24.2)

**Part 4:** *Model Deployment and Integration*

* FastAPI (Latest version: 0.100.1)
* Pydantic (Latest version: 2.1.1)
* Uvicorn (Latest version: 0.23.1)
* Docker

## Acquiring The Results of Part 1, 2 and 3 Separately

By following these steps, you can check the progress and result of part 1, 2 and 3 separately, from data acquisition, data analysis to the machine learning model training and obtaining its accuracy score.

**Part 1:** *Data Acquisition (Immo Scraping)*

1. Navigate to immo_scraping folder.
2. Run dataset_immo_scrapper.py to acquire the dataset (dataset-immo.csv).

**Part 2:** *Data Cleaning, Analysis, and Interpretation (Data Exploration)*

1. Go to data_exploration folder.
2. Open and run notebook-immo-analysis.ipynb for data analysis and insights.

**Part 3:** *Machine Learning with Random Forest Regressor*

1. Navigate to the model_building folder;
2. Run the main script for machine learning and model evaluation by executing the file main.py.
3. The final accuracy score of the model will be displayed on the terminal;
4. The model is storaged in a picke file that can be further loaded to make predictions.

## Acknowledgments

The Property Value Estimation project was implemented during the period of 27/06/2023 until 28/07/2023 by Andreia Azevedo Heringer Negreira as part of the AI Boocamp at BeCode.org. The code is provided as an educational example and can be modified and expanded upon to suit different requirements and preferences.

### Contributor Contact

Andreia Azevedo Heringer Negreira can be reached by: [Github](https://github.com/andreia-negreira) and [LinkedIn](https://www.linkedin.com/in/andreia-heringer-negreira-0b6a2a13a/).

---

Unlock the magic of property predictions with the enchanting Property Value Estimation App - making real estate decisions has never been this bewitching! :european_castle::crystal_ball::sparkles:
