from audioop import add
import streamlit as st

import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.figure_factory as ff

COLOR_1 = '#F72585'
COLOR_2 = '#7209B7'
COLOR_3 = '#3A0CA3'
COLOR_4 = '#4361EE'
COLOR_5 = '#4CC9F0'
colors = [COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5]


df_train = pd.read_csv("./spaceship-titanic-data/train.csv")
df_test = pd.read_csv("./spaceship-titanic-data/test.csv")



add_sidebar = st.sidebar.selectbox('Part of data discovery', ('Train Dataset','Missing Values', 'EDA'))

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

elif add_sidebar == 'EDA':
    st.header('Exploratory Data Analysis')

    st.subheader('Transported distribution')
    fig = px.pie(data_frame = df_train, names = 'Transported',color_discrete_sequence = colors, hole = 0.6, opacity=0.8)
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

    col1, col2 = st.columns(2)
    
    columns = [col1, col2]
    for i, var_name in enumerate(exp_feats):

        with columns[0]:
            fig = px.histogram(data_frame = df_train, x = var_name, color = 'Transported',color_discrete_sequence = colors, nbins = 30, title = var_name, opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)

        with columns[1]:
            q_hi  = df_train[var_name].quantile(0.75)
            Age_true = df_train[(df_train.Transported == True) and (df_train[var_name] <= q_hi)][var_name].dropna().tolist()
            Age_false = df_train[(df_train.Transported == False) and (df_train[var_name] <= q_hi)][var_name].dropna().tolist()
            hist_data = [Age_true, Age_false]

            fig2 = ff.create_distplot(hist_data=hist_data, group_labels=group_labels, bin_size=[1, 1], show_rug=False, colors=colors)
            st.plotly_chart(fig2, use_container_width=True)
        
        
    
"""# Right plot (truncated)
        ax=fig.add_subplot(5,2,2*i+2)
        sns.histplot(data=train, x=var_name, axes=ax, bins=30, kde=True, hue='Transported')
        plt.ylim([0,100])
        ax.set_title(var_name)"""