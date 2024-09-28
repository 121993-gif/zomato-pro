import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# Sidebar for page selection
select_page = st.sidebar.radio('Select page', ['Introduction', 'Analysis', 'Model Classification'])

# Introduction Page
if select_page == 'Introduction':
    def main():
        st.title('Zomato Bangalore Restaurants')
        st.write('### Introduction to the Data:')
        st.write('''In the vibrant culinary landscape of Bangalore, Zomato serves as a crucial platform for food lovers, offering insights into a diverse array of restaurants. This project aims to analyze restaurant data from Zomato, focusing on key metrics such as customer ratings, cuisines, price ranges, and geographic distribution. By leveraging data analytics and visualization techniques, we seek to uncover trends, popular dining spots, and customer preferences in Bangalore's dynamic food scene. The insights derived from this analysis can provide valuable recommendations for both consumers and restaurant owners, enhancing the overall dining experience in the city.''')
        
        st.header('Dataset Feature Overview')
        st.write('''
            *url*: contains the URL of the restaurant on the Zomato website.
            *address*: contains the address of the restaurant in Bengaluru.
            *name*: contains the name of the restaurant.
            *online_order*: whether online ordering is available in the restaurant or not.
            *book_table*: table booking option available or not.
            *rate*: contains the overall rating of the restaurant out of 5.
            *votes*: contains total number of ratings for the restaurant as of the above mentioned date.
            *phone*: contains the phone number of the restaurant.
            *location*: contains the neighborhood in which the restaurant is located.
            *rest_type*: restaurant type.
            *dish_liked*: dishes people liked in the restaurant.
            *cuisines*: food styles, separated by commas.
            *approx_cost(for two people)*: contains the approximate cost for a meal for two people.
            *reviews_list*: list of tuples containing reviews for the restaurant.
            *menu_item*: contains list of menus available in the restaurant.
            *listed_in(type)*: type of meal.
            *listed_in(city)*: contains the neighborhood in which the restaurant is listed.
        ''')

    if __name__ == '__main__':
        main()

# Analysis Page
elif select_page == 'Analysis':
    def main():
        try:
            cleaned_df = pd.read_csv('cleaned_df.csv')
        except FileNotFoundError:
            st.error("File 'cleaned_df.csv' not found. Please ensure it's in the correct directory.")
            return

        st.write('### Head of Dataframe')
        st.dataframe(cleaned_df.head(10))
        
        tab1, tab2, tab3 = st.tabs(['Univariate Analysis', 'Bivariate Analysis', 'Multivariate Analysis'])

        # Univariate Analysis
        tab1.write('### Univariate Analysis with Histogram for Each Feature')
        for col in cleaned_df.columns:
            tab1.plotly_chart(px.histogram(cleaned_df, x=col))

        # Bivariate Analysis
        tab2.write('### What is the correlation between cost_category and rate?')
        tab2.plotly_chart(px.box(cleaned_df, x='cost_category', y='rate'))

        tab2.write('### How do votes relate to rate?')
        tab2.plotly_chart(px.bar(cleaned_df, x='vote_category', y='rate'))

        avg_ratings = cleaned_df.groupby('online_order')['rate'].mean().reset_index()
        tab2.write('### How do average ratings differ between restaurants that offer online order vs. those that don’t?')
        tab2.plotly_chart(px.bar(avg_ratings, x='online_order', y='rate'))

        tab2.write('### Is there a significant difference in ratings based on restaurant type or cuisines?')
        tab2.plotly_chart(px.box(cleaned_df, x='rest_type', y='rate'))
        tab2.plotly_chart(px.box(cleaned_df, x='cuisines', y='rate'))

        tab2.write('### How do average ratings vary by location_city or dining_type?')
        avg_ratings_city = cleaned_df.groupby('location_city')['rate'].mean().reset_index()
        avg_ratings_dining = cleaned_df.groupby('dining_type')['rate'].mean().reset_index()

        tab2.subheader("Average Ratings by Location City")
        tab2.plotly_chart(px.bar(avg_ratings_city, x='location_city', y='rate'))
        
        tab2.subheader("Average Ratings by Dining Type")
        tab2.plotly_chart(px.bar(avg_ratings_dining, x='dining_type', y='rate'))

        tab3.write('### How do ratings vary by the type of cuisine and cost_category?')
        tab3.plotly_chart(px.scatter(cleaned_df, x='cuisines', y='rate', color='cost_category'))

        tab3.write('### What is the relationship between approximate cost and ratings?')
        tab3.plotly_chart(px.scatter_3d(cleaned_df, x='cost_category', y='rate', z='rate', color='cuisines'))

        tab3.write('##### Do dining types affect the relationship between approximate cost and ratings?')
        tab3.plotly_chart(px.scatter(cleaned_df, x='cost_category', y='rate', color='dining_type'))

        tab3.write('##### How does the combination of online ordering and table booking options relate to ratings and approximate costs?')
        cleaned_df['options'] = cleaned_df['online_order'].astype(str) + ' & ' + cleaned_df['book_table'].astype(str)
        tab3.plotly_chart(px.scatter(cleaned_df, x='cost_category', y='rate', color='options'))

        tab3.write('##### Is there a difference in ratings based on city, given the approximate cost?')
        tab3.plotly_chart(px.box(cleaned_df, x='location_city', y='rate', color='cost_category'))

        tab3.write('##### How do average ratings differ between high-cost and low-cost restaurants across various cuisines?')   
        avg_ratings_hl = cleaned_df.groupby(['cuisines', 'cost_category'])['rate'].mean().reset_index()
        tab3.plotly_chart(px.bar(avg_ratings_hl, x='cuisines', y='rate', color='cost_category'))


    if __name__ == '__main__':
        main()

# Model Classification Page
elif select_page == 'Model Classification':
    def main(): 
        st.title('Model Classification')
        pipeline = joblib.load('RF_pipeline.pkl')

        def Prediction(url, address, name, online_order, book_table, rate, location, rest_type, dish_liked, cuisines, menu_item, dining_type, location_city, cost_category, online_booking_combined, vote_category):
            df = pd.DataFrame(columns=['url', 'address', 'name', 'online_order', 'book_table', 'rate', 'location', 'rest_type', 'dish_liked', 'cuisines', 'menu_item', 'dining_type', 'location_city', 'cost_category', 'online_booking_combined', 'vote_category'])
            df.at[0, 'url'] = url
            df.at[0, 'address'] = address
            df.at[0, 'name'] = name
            df.at[0, 'online_order'] = online_order
            df.at[0, 'book_table'] = book_table
            df.at[0, 'rate'] = float(rate)
            df.at[0, 'location'] = location
            df.at[0, 'rest_type'] = rest_type
            df.at[0, 'dish_liked'] = dish_liked
            df.at[0, 'cuisines'] = cuisines
            df.at[0, 'menu_item'] = menu_item
            df.at[0, 'dining_type'] = dining_type
            df.at[0, 'location_city'] = location_city
            df.at[0, 'cost_category'] = cost_category
            df.at[0, 'online_booking_combined'] = online_booking_combined
            df.at[0, 'vote_category'] = vote_category

            result = pipeline.predict(df)[0]
            return result

        # User Input
        url = st.text_input('Please write the restaurant URL')
        address = st.text_input('Please write your address')
        name = st.text_input('Please write your name')
        online_order = st.selectbox('Is online ordering available?', ['Yes', 'No'])
        book_table = st.selectbox('Is table booking option available?', ['Yes', 'No'])
        rate = st.text_input('Please write your rating')
        location = st.text_input('Please write your location')
        rest_type = st.text_input('Please write your restaurant type')
        dish_liked = st.text_input('Please write the dish liked')
        cuisines = st.text_input('Please write your cuisines')
        menu_item = st.text_input('Please write your menu items')
        dining_type = st.text_input('Please write your dining type') 
        location_city = st.text_input('Please write your location city')
        cost_category = st.selectbox('Please select your cost category', ['High Cost', 'Low Cost'])
        online_booking_combined = st.selectbox('Combine online order and booking?', ['Yes', 'No'])
        vote_category = st.text_input('Please provide the vote category')

        if st.button('Predict'):
            result = Prediction(url, address, name, online_order, book_table, rate, location, rest_type, dish_liked, cuisines, menu_item, dining_type, location_city, cost_category, online_booking_combined, vote_category)
            st.write('### Prediction Result:')
            st.write(f'The predicted result is: {round(np.exp(result), 2)}')

    if __name__ == '__main__':
        main()
