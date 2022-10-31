from audioop import add
from turtle import title
import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)

import feature_engineering as fe

COLOR_1 = '#F72585'
COLOR_2 = '#7209B7'
COLOR_3 = '#3A0CA3'
COLOR_4 = '#4361EE'
COLOR_5 = '#4CC9F0'
colors = [COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5]

opacity = 0.8


df_train = pd.read_csv("./spaceship-titanic-data/train.csv")
df_test = pd.read_csv("./spaceship-titanic-data/test.csv")



add_sidebar = st.sidebar.selectbox('Part of data discovery', ('Train Dataset','Missing Values', 'Exploratory Data Analysis', 'Feature Engineering'))

st.title('🧑‍🚀 Spaceship Titanic')

if add_sidebar == 'Train Dataset':
    st.header('Train Dataset')
    st.write(df_train)

    """
    **Feature descriptions :**

    - **PassengerId** - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group.People in a group are often family members, but not always.

    - **HomePlanet** - The planet the passenger departed from, typically their planet of permanent residence.

    - **CryoSleep** - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.

    - **Cabin** - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.

    - **Destination** - The planet the passenger will be debarking to.

    - **Age** - The age of the passenger.

    - **VIP** - Whether the passenger has paid for special VIP service during the voyage.

    - **RoomService, FoodCourt, ShoppingMall, Spa, VRDeck** - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.

    - **Name** - The first and last names of the passenger.

    - **Transported** - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.
    """

elif add_sidebar == 'Missing Values':
    st.header('Missing Values')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Train set")
        st.write(df_train.isna().sum())

    with col2:
        st.subheader("Test set")
        st.write(df_test.isna().sum())

elif add_sidebar == 'Exploratory Data Analysis':
    st.header('Exploratory Data Analysis')

    st.subheader('Transported distribution')
    fig = px.pie(data_frame = df_train, names = 'Transported',color_discrete_sequence = colors, hole = 0.6, opacity=opacity)
    st.plotly_chart(fig, use_container_width=True)
    
    # Group data together
    st.subheader('Age distribution')
    Age_true = df_train[df_train.Transported == True]['Age'].dropna().tolist()
    Age_false = df_train[df_train.Transported == False]['Age'].dropna().tolist()
    hist_data = [Age_true, Age_false]

    group_labels = ['Transported', 'Not transported']

    # Create distplot with custom bin_size
    fig2 = ff.create_distplot(hist_data=hist_data, group_labels=group_labels, bin_size=[1, 1], show_rug=False, colors=colors)
    st.plotly_chart(fig2, use_container_width=True)

    """
    **Notes :**

    - 0-18 year olds were more likely to be transported than not.
    - 18-25 year olds were less likely to be transported than not.
    - Over 25 year olds were about equally likely to be transported than not.
    
    **Insight :**

    Create a new feature that indicates whether the passanger is a child, adolescent or adult.
    """

    st.subheader('Expenditure features')

    exp_feats=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # Plot expenditure features

    tab1, tab2, tab3, tab4, tab5 = st.tabs(exp_feats)
    tabs = [tab1, tab2, tab3, tab4, tab5]

    for i, var_name in enumerate(exp_feats):

        with tabs[i]:
            # inverser les  couleurs de true et false ici
            fig = px.histogram(data_frame = df_train, x = var_name, color = 'Transported',color_discrete_sequence = colors, nbins = 30, opacity=opacity, title=var_name + ' distribution')
            st.plotly_chart(fig, use_container_width=True)

            # inverser les  couleurs de true et false ici
            fig2 = px.histogram(data_frame = df_train, x = var_name, color = 'Transported',color_discrete_sequence = colors, nbins = 30, opacity=opacity, log_y=True, title=var_name + ' log distribution')
            st.plotly_chart(fig2, use_container_width=True)

    """
    **Notes :**

    - Most people don't spend any money.

    - The distribution of spending decays exponentially.

    - There are a small number of outliers.

    - People who were transported tended to spend less.

    - RoomService, Spa and VRDeck have different distributions to FoodCourt and ShoppingMall - we can think of this as luxury vs essential amenities.

    **Insight :**

    - Create a new feature that tracks the total expenditure across all 5 amenities.

    - Create two features for luxury and essential amenities.
    
    - Create a binary feature to indicate if the person has not spent anything. (i.e. total expenditure is 0).
    
    - Take the log transform to reduce skew.
    """

    st.subheader('Categorical features')

    # Categorical features
    cat_feats=['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
    tab1, tab2, tab3, tab4 = st.tabs(cat_feats)
    tabs = [tab1, tab2, tab3, tab4]

    # Plot categorical features
    fig=plt.figure(figsize=(10,16))
    for i, var_name in enumerate(cat_feats):

        with tabs[i]:
            fig = px.histogram(data_frame = df_train, x = var_name, color = 'Transported',color_discrete_sequence = colors, opacity=opacity, barmode='group')
            st.plotly_chart(fig, use_container_width=True)

    """
    **Notes :**

    - VIP does not appear to be a useful feature, the target split is more or less equal.
    
    - CryoSleep appears the be a very useful feature in contrast.
    
    **Insights :**

    We might consider dropping the VIP column to prevent overfitting.
    """

    st.subheader('Qualitative features')
    # Qualitative features
    qual_feats=['PassengerId', 'Cabin' ,'Name']

    # Preview qualitative features
    st.write(df_train[qual_feats].head())

    """
    **Notes :**

    - PassengerId takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group.
    
    - Cabin takes the form deck/num/side, where side can be either P for Port or S for Starboard.
    
    **Insights :**

    - We can extract the group and group size from the PassengerId feature.
    
    - We can extract the deck, number and side from the cabin feature.
    
    - We could extract the surname from the name feature to identify families.
    """

elif add_sidebar == 'Feature Engineering':
    st.header('Feature Engineering')

    st.subheader('Age group')

    df_train = fe.create_ageGroup(df_train)
    df_test = fe.create_ageGroup(df_test)

    order_age_group = ['Age_0-12','Age_13-17','Age_18-25','Age_26-30','Age_31-50','Age_51+']

    fig = px.histogram(data_frame = df_train, x = 'Age_group', color = 'Transported',color_discrete_sequence = colors, opacity=opacity, barmode='group', title='Age Group Distribution',category_orders={'Age_group': order_age_group})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Expenditure')

    df_train = fe.create_expenditure(df_train)
    df_test = fe.create_expenditure(df_test)

    fig = px.histogram(data_frame = df_train, x = 'Expenditure', color = 'Transported',color_discrete_sequence = colors, nbins = 60, opacity=opacity, log_y=True, title='Expenditure log distribution')
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(data_frame = df_train, x = 'No_spending', color = 'Transported',color_discrete_sequence = colors, opacity=opacity, barmode='group', title='No Spending Indicator')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Passenger Group')

    df_train, df_test = fe.create_passengerGroup(df_train, df_test)

    fig = px.histogram(data_frame = df_train, x = 'Group_size', color = 'Transported',color_discrete_sequence = colors, opacity=opacity, barmode='group', title='Group Size')
    st.plotly_chart(fig, use_container_width=True)

    """
    We can't really use the Group feature in our models because it has too big of a cardinality (6217) and would explode the number of dimensions with one-hot encoding.

    The Group size on the other hand should be a useful feature. In fact, we can compress the feature further by creating a 'Solo' column that tracks whether someone is travelling on their own or not. The figure on the right shows that group size=1 is less likely to be transported than group size>1.
    """

    st.subheader('Cabin Location')

    