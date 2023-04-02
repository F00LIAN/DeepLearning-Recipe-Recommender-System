# NATURAL LANGUAGE PROCESSING Food Recommendation System 
### This project explores two different recommendation systems: Traditional Systems and Deep Learning Transformers Recommendation System for food recommendation.

## The Data 
### The data used in this project comes from the Food.com Kaggle Dataset, which includes two Parquet files with user reviews and food dish labels with ingredients.
### The Data comes from the Food.com Kaggle Dataset. https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews 

## Part 1 - Collaborive vs Content Based Filtering for System
### Two different recommendation systems were explored in this project: Collaborative Filtering and Content-Based Filtering. Popularity and User-Based approaches were also analyzed.

## Part 2 - Building the recommendation system using traditional methods. 
### In this section, a recommendation system was built using traditional methods. Cosine Similarity was used to find distances, and a function was defined to determine the similarity score based on the closest distances. However, one of the limitations of this method is that users have to input the direct title/name of the dish according to the original dataset, and some dishes may not be very related to the recommendation input. 

## Part 3 - Recommender System using modern methods
### In this section, modern methods were employed to build the recommendation system. The HuggingFace Library was used to extract embeddings, and Transformers were trained on massive amounts of data. These models can now be used for smaller downstream tasks, such as food recommendation. One of the benefits of this method is that context is captured better, and the system can integrate. PCA was used to visualize a 2D representation of how the Nearest Neighbors algorithm will find distances.

## Next Steps
### The next step is to integrate a chatbot along with a frontend UI/Streamlit application to make the system more user-friendly and accessible.