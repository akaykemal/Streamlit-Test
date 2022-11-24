import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

st.markdown(
    '''
    <style>
    .main {
    background-color: #8b00b8;
    }
    </style
    ''',
    unsafe_allow_html=True    
)

@st.cache
def get_data(filename):
    data = pd.read_csv(filename)
    return data

with header:
    st.title('Welcome to my awesome data science project')
    st.text('In this project I look into the transaction of taxis in NYC. ...')

with dataset:
    st.header('NYC taxi dataset')
    st.text('I found this dataset on blabla.com')

    taxi_data = get_data('./data/taxi_data.csv')
    st.write(taxi_data.head())

    #st.write(taxi_data[LocationID])

    st.subheader('Sub header fotr the zooones')
    zones = pd.DataFrame(taxi_data["Zone"].value_counts()).head(20)
    st.bar_chart(zones)

with features:
    st.header('The features I created')

    st.markdown('* **first feature: I created this feature bc of this.. bc of this logic..**')
    st.markdown('* **first feature: I created this feature bc of this.. bc of this logic..**')

with modelTraining:
    st.header('Time to train the model')
    st.text('Here you get to choose fdgsdfg sdfkgj sdflkgj sdfgl kjsdfkl gjlksd asdfsadfsadf asdf ')

    sel_col, disp_col =st.columns(2)

    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100,200,300,'No Limit'],index=0)

    sel_col.text('list of data')
    sel_col.write(taxi_data.columns)
    
    input_feature = sel_col.text_input('Which feature should be used as the input feature?','LocationID')

    if n_estimators == 'No Limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    x=taxi_data[[input_feature]]
    y=taxi_data[['LocationID']]

    regr.fit(x,y)
    prediction=regr.predict(y)

    disp_col.subheader('MAE:')
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader('MSE:')
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R2:')
    disp_col.write(r2_score(y, prediction))
