'''
Author: Jeffrey Sultan
Date: 8/4/22
Dataset: Uber Fares Data csv
App URL: ###############
Description: This interactive application containing Uber ride data from 2009 to 2015.
            The data is depicted in a series of visualizations to illustrate various metrics that
            were found. The original data table was updated to remove unneeded information. Date,
            time and straight-line calculated distance between pickup and drop-off location was
            also added to the data table.

'''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import geopy.distance
import seaborn as sns
import matplotlib.pyplot as plt

# Set constants
ARRIVAL_DATE_COLUMN = 'pickup_datetime'
DATA = "UberData.csv"
#---------------------------------------------------------------------
# Code checklist: Function: function w/ defualt parameter, two parameters, returns value
def getData(key, dict='m'):
    monthDict = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"Decemeber"}
    weekTpl = ("Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday")

    if dict == 'm':
        return monthDict[key]
    elif dict == 'w':
        return weekTpl[key]
#---------------------------------------------------------------------
# Set page layout and create header
st.set_page_config(layout='wide')

# Initialize application with title, header and descriptive information
# st cols(#1, #2) where #1 = width of col1 and #2 = width of col2
row0_1, row0_2 = st.columns((3,1))
with row0_1:
    st.title('Uber pickups in NYC')
with row0_2:
    st.text("")
    st.subheader('Streamlit App created by:')
    st.text("Jeffrey Sultan")
# col1 spacer = 0.1 padding, col2 width, col3 spacer = 0.1 padding
row3_spacer1, row3_1, row3_spacer2 = st.columns((0.1, 1, 0.1))
with row3_1:
    st.markdown("This interactive application containing Uber ride data from 2009 to 2015. The data is depicted in a series of visualizations to illustrate various metrics that were found. ")
    st.markdown("The original data table was updated to remove unneeded information. ")
    st.markdown("Date, time and straight-line calculated distance between pickup and drop-off location was also added to the data table. ")
st.title('')

# creating a sidebar to display additional project information
# Code checklist - At least one function that does not return a value
def load_sidebar():
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This application uses the following charts to visualize the data collected: 
        \n1) A map of the pickup location 
        \n2) A scatter plot of fare vs distance 
        \n3) A pie chart of the date/time distribution 
        \n4) A line chart of the pickup time distribution
        \n5) A scatter plot with linear prediction 
        """
    )
    st.sidebar.title("Author - Date:")
    st.sidebar.info("Author: Jeffrey Sultan")
    st.sidebar.info("Date: 04Aug2022")
    st.sidebar.info("Dataset: Uber Fares Data CSV")
load_sidebar()
#------------------------------------------------------------------------
# Section 0 - Start with loading data from the CSV

def load_data():
    data = pd.read_csv(DATA)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)

    return data

# Loading bar used to work but for some reason i no longer see it
data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("")
#------------------------------------------------------------------------
# Section 1 - Reframe the data provided to be more accessible and removing outliers

# The line below reframes the pickup and dropoff locations:
# Latitude should be 40 to 41. Longitude should be -73 to -75
data = data[(data.pickup_latitude<41) & (data.dropoff_latitude<41) &
        (data.pickup_latitude>40) & (data.dropoff_latitude>40) &
        (data.pickup_longitude<-73) & (data.dropoff_longitude<-73) &
        (data.pickup_longitude>-75) & (data.dropoff_longitude>-75)]

# Renaming the pickup latitude and longitude to work with map utility
data = data.rename({'pickup_longitude': 'longitude', 'pickup_latitude': 'latitude'}, axis=1)
data[ARRIVAL_DATE_COLUMN] = pd.to_datetime(data[ARRIVAL_DATE_COLUMN])

# Remove unnecessary columns (Unnamed and key columns)
# Code checklist: dataframe: drop columns
data.drop(['unnamed: 0', 'key'], axis=1, inplace=True)

# Create new columns to split the arrival date column from date/time conversion
# Code checklist: dataframe: add columns
data['year'] = data.pickup_datetime.dt.year
data['month'] = data.pickup_datetime.dt.month
data['weekday'] = data.pickup_datetime.dt.weekday
data['hour'] = data.pickup_datetime.dt.hour

# Using the geopy module, we can create a new column that evaluates the distance between pickup and drop off locations.
# Note that the pickup lat/long columns were renamed above from 'pickup_latitude' to 'latitude' and same for longitude
# geopy.distance.distance uses the geodesic distance which is the shortest distance on the surface of an ellipsoidal model of the earth (Karney 2013)
# The calculated distance is in meters, rounded to 2 decimal places and placed in a new column titled 'distance'
data['distance']=[round(geopy.distance.distance((data.latitude[i], data.longitude[i]),(data.dropoff_latitude[i], data.dropoff_longitude[i])).m,2) for i in data.index]

# Reframe the data set to exclude fare_amount outliers more than $100 and distance outliers less than 1000 meters
data = data[(data.fare_amount<=100)]
data = data[(data.distance > 1000)]

#------------------------------------------------------------------------
# Section 2 - Option to show the reframed data from above + stats

# Checkbox to show raw data
# Code checklist: pandas: sort column asc/desc
if st.checkbox('Show data table'):
    st.subheader('Reframed data with added columns')
    fare_median = data['fare_amount'].median()
    dist_median = data['distance'].median()
    st.text(f'{len(data)} Entries\t ${fare_median} Median of Ride Fares\t {dist_median}m Median of Distances Travelled')
    st.write(data.sort_values('distance', ascending=False))
#------------------------------------------------------------------------
# Section 3 - Visualize the pickup locations by hour on a map of NYC
# Separation bar html copied from streamlit discussion bar - see references
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
st.subheader('| VISUALIZING MAP OF ALL PICKUP LOCATIONS DEPENDENT ON MONTH OR HOUR')
map_option = st.radio(
     "How to filter the data appearing on the map?",
     ('Pickup Hour', 'Pickup Month', 'Pickup Weekday'))

# Code checklist: Data structures: Dictionary & Streamlit UI Controls: Slider
# Code checklist: At least one function that is called twice
if map_option == 'Pickup Month':
    # Sliding bar for mapping feature: min = (1-Jan), max = (12-Dec)
    month_to_filter = st.slider('MONTH OF THE YEAR', 1, 12)
    selected_month = getDict(month_to_filter, 'm')
    filtered_data = data[data[ARRIVAL_DATE_COLUMN].dt.month == month_to_filter]
    st.subheader(f"Map of all pickups in {selected_month}")
    st.text(f"Showing {len(filtered_data)} pickups")
    st.map(filtered_data)
elif map_option == 'Pickup Weekday':
    # Sliding bar for mapping feature: min = 0hr, max = 23hr, default = 17hr
    weekday_to_filter = st.slider('DAY OF WEEK', 0, 6)
    selected_weekday = getData(weekday_to_filter, 'w')
    filtered_data = data[data[ARRIVAL_DATE_COLUMN].dt.weekday == weekday_to_filter]
    st.subheader(f'Map of all pickups on {selected_weekday}')
    st.text(f"Showing {len(filtered_data)} pickups")
    st.map(filtered_data)
elif map_option == 'Pickup Hour':
    # Sliding bar for mapping feature: min = 0hr, max = 23hr, default = 17hr
    hour_to_filter = st.slider('HOUR OF THE DAY', 0, 23, 17)
    filtered_data = data[data[ARRIVAL_DATE_COLUMN].dt.hour == hour_to_filter]
    st.subheader('Map of all pickups at %s:00' % hour_to_filter)
    st.text(f"Showing {len(filtered_data)} pickups")
    st.map(filtered_data)

#------------------------------------------------------------------------
# Section 4 - Scatter Plot visualization of fares vs distance with optional filters
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
st.subheader('| VISUALIZING SCATTERPLOT OF FARES VS DISTANCE AND VARIABLE HUE')

fig4_option = st.selectbox('USE VARIABLE TO SET HUE',('month', 'year', 'weekday', 'hour'))
fig4 = plt.figure()
# next line somehow controls grid lines in my graph. Copied from a seaborn discussion on stack exchange. see refs
sns.set(rc={'figure.figsize':(8,6)})
plt.ylim(0, 100)

plt.title('Fares vs Distance')
plt.xlabel('Straight-line Distance (meters)')
plt.ylabel('Fare ($)')

# Scatter plot created using the seaborn module
sns.scatterplot(data=data, x='distance',y='fare_amount',hue=fig4_option, s=10)
st.pyplot(fig4)

# Next create a pie chart to illustrate the percentage of entries found for each hue option selected above
if fig4_option == 'year':
    ind1 = pd.DataFrame(data.groupby('year').year.count()).rename(columns={'year': 'count'}).reset_index()
    g1 = px.pie(ind1,
                values='count',
                names='year',
                color='year',
                color_discrete_map={'2009': 'red', '2010': 'blue', '2011': 'green', '2012': 'yellow', '2013': 'orange',
                                    '2014': 'purple', '2015': 'brown'},
                title='| PERCENTAGE OF DATA RECORDED FROM EACH YEAR')
    g1.update_traces(textposition='inside',
                     textinfo='percent+label',
                     showlegend=False)
    st.plotly_chart(g1, use_container_width=True)
elif fig4_option == 'month':
    ind1 = pd.DataFrame(data.groupby('month').month.count()).rename(columns={'month': 'count'}).reset_index()
    g1 = px.pie(ind1,
                values='count',
                names='month',
                color='month',
                title='| PERCENTAGE OF DATA RECORDED FROM EACH MONTH')
    g1.update_traces(textposition='inside',
                     textinfo='percent+label',
                     showlegend=False)
    st.plotly_chart(g1, use_container_width=True)
elif fig4_option == 'weekday':
    ind1 = pd.DataFrame(data.groupby('weekday').weekday.count()).rename(columns={'weekday': 'count'}).reset_index()
    g1 = px.pie(ind1,
                values='count',
                names='weekday',
                color='weekday',
                title='| PERCENTAGE OF DATA RECORDED FROM EACH WEEKDAY')
    g1.update_traces(textposition='inside',
                     textinfo='percent+label',
                     showlegend=False)
    st.plotly_chart(g1, use_container_width=True)
elif fig4_option == 'hour':
    ind1 = pd.DataFrame(data.groupby('hour').hour.count()).rename(columns={'hour': 'count'}).reset_index()
    g1 = px.pie(ind1,
                values='count',
                names='hour',
                color='hour',
                title='| PERCENTAGE OF DATA RECORDED FROM EACH HOUR')
    g1.update_traces(textposition='inside',
                     textinfo='percent+label',
                     showlegend=False)
    st.plotly_chart(g1, use_container_width=True)

#------------------------------------------------------------------------
# Section 5 - Chart representing the data from each year and time of day
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
st.subheader('| VISUALIZING DATA OF RELATED TO DATE & TIME INFORMATION')

# Bar chart that illustrates the amount of data recorded from each time of day
fig5_2_option = st.selectbox('USE VARIABLE TO FILTER DATA', ('year', 'month'))
if fig5_2_option == 'year':
    fig5_year_to_filter = st.slider('YEAR', 2009, 2015)
    filtered_data = data[data[ARRIVAL_DATE_COLUMN].dt.year == fig5_year_to_filter]
elif fig5_2_option == 'month':
    fig5_month_to_filter = st.slider('MONTH', 1, 12)
    filtered_data = data[data[ARRIVAL_DATE_COLUMN].dt.month == fig5_month_to_filter]
ind2 = pd.DataFrame(filtered_data.groupby('hour').hour.count()).rename(columns={'hour':'count'}).reset_index()
g3 = px.line(ind2,
            x='hour',
            y='count',
            title='| POPULAR PICKUP TIMES EACH DAY')
st.plotly_chart(g3, use_container_width=True)
#------------------------------------------------------------------------
# Section 6 - Scatter plot of distance vs fare amount and prediction of fare based on distance
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
st.subheader('| LINEAR REGRESSION OF DISTANCE VS FARE AMOUNT')

# Manual calculation of Y-estimate using Ordinary Least Squares (could not get stats package to work correctly)
x = filtered_data['distance']
y = filtered_data['fare_amount']

# Calculate the mean of X and y
xmean = np.mean(x)
ymean = np.mean(y)

# Calculate the terms needed for the numerator and denominator of beta
df_temp = pd.DataFrame()
df_temp['xycov'] = (x - xmean) * (y - ymean)
df_temp['xvar'] = (y - xmean)**2

# Calculate beta and alpha
beta = df_temp['xycov'].sum() / df_temp['xvar'].sum()
alpha = ymean - (beta * xmean)

# Integrate user interaction to predict fare amount vs distance
selected_distance = st.slider("DISTANCE IN METERS", 0, 50000, 10000)
y_prediction_from_selected = alpha + beta * selected_distance
st.write('Estimated Fare for Selected Distance = \\$', round(y_prediction_from_selected, 2))

# Plot manual calculation against actual data
ypred = alpha + beta * x
fig7 = plt.figure(figsize=(12, 6))
plt.plot(x, ypred)     # regression line
plt.plot(x, y, 'ro')   # scatter plot showing actual data
plt.plot(selected_distance, y_prediction_from_selected, marker="o", markersize=15, markerfacecolor="blue")
plt.title('Linear Regression Based on Distance')
plt.xlabel('Distance (meters)')
plt.ylabel('Fare Amount ($)')

# Plot the regression line against actual distance vs fare data
st.pyplot(fig7)
#-----------------------------------------------------------------------------------
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
st.subheader('| ONLINE EXAMPLES REFERENCED')

st.write("https://towardsdatascience.com/introduction-to-linear-regression-in-python-c12a072bedf0")
st.write("https://www.kaggle.com/code/yasserh/uber-fare-prediction-comparing-best-ml-models")
st.write("https://pythonwife.com/pie-chart-with-plotly/")
st.write("https://seaborn.pydata.org/tutorial/regression.html")
st.write("https://github.com/tdenzl/BuLiAn")
st.write("https://discuss.streamlit.io/t/horizontal-separator-line/11788")
st.write("https://stackoverflow.com/questions/31594549/how-to-change-the-figure-size-of-a-seaborn-axes-or-figure-level-plot")