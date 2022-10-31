import numpy as np
import pandas as pd

def create_ageGroup(df):
    df['Age_group']=np.nan
    df.loc[df['Age']<=12,'Age_group']='Age_0-12'
    df.loc[(df['Age']>12) & (df['Age']<18),'Age_group']='Age_13-17'
    df.loc[(df['Age']>=18) & (df['Age']<=25),'Age_group']='Age_18-25'
    df.loc[(df['Age']>25) & (df['Age']<=30),'Age_group']='Age_26-30'
    df.loc[(df['Age']>30) & (df['Age']<=50),'Age_group']='Age_31-50'
    df.loc[df['Age']>50,'Age_group']='Age_51+'

    return df

def create_expenditure(df):
    exp_feats=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    df['Expenditure']=df[exp_feats].sum(axis=1)
    df['No_spending']=(df['Expenditure']==0).astype(int)
    return df

def create_passengerGroup(df_train, df_test):
    df_train['Group'] = df_train['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
    df_test['Group'] = df_test['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
    df_train['Group_size']=df_train['Group'].map(lambda x: pd.concat([df_train['Group'], df_test['Group']]).value_counts()[x])
    df_test['Group_size']=df_test['Group'].map(lambda x: pd.concat([df_train['Group'], df_test['Group']]).value_counts()[x])
    df_train['Solo']=(df_train['Group_size']==1).astype(int)
    df_test['Solo']=(df_test['Group_size']==1).astype(int)

    return df_train, df_test

def create_cabinLocation(df):
    # Replace NaN's with outliers for now (so we can split feature)
    df['Cabin'].fillna('Z/9999/Z', inplace=True)

    # New features
    df['Cabin_deck'] = df['Cabin'].apply(lambda x: x.split('/')[0])
    df['Cabin_number'] = df['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
    df['Cabin_side'] = df['Cabin'].apply(lambda x: x.split('/')[2])

    # Put Nan's back in (we will fill these later)
    df.loc[df['Cabin_deck']=='Z', 'Cabin_deck']=np.nan
    df.loc[df['Cabin_number']==9999, 'Cabin_number']=np.nan
    df.loc[df['Cabin_side']=='Z', 'Cabin_side']=np.nan

    # Drop Cabin (we don't need it anymore)
    df.drop('Cabin', axis=1, inplace=True)

    return df