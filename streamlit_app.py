import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Set page config
st.set_page_config(
    page_title="Netflix Content Analysis Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #E50914;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 2rem;
        color: #E50914;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #E50914;
        padding-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data():
    """Load and clean the Netflix dataset"""
    try:
        df = pd.read_csv("netflix_titles.csv")
        
        # Data cleaning
        df['director'].fillna('Unknown', inplace=True)
        df['cast'].fillna('Unknown', inplace=True)
        df['country'].fillna('Unknown', inplace=True)
        df.dropna(subset=['date_added', 'rating'], inplace=True)
        
        # Date processing
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        df['year_added'] = df['date_added'].dt.year
        df['month_added'] = df['date_added'].dt.month
        
        # Duration processing
        df[['duration_int', 'duration_type']] = df['duration'].str.extract(r'(\d+)\s+(\w+)')
        df['duration_int'] = pd.to_numeric(df['duration_int'], errors='coerce')
        
        # Genre processing
        df['main_genre'] = df['listed_in'].apply(lambda x: x.split(',')[0])
        df['primary_country'] = df['country'].apply(lambda x: x.split(',')[0] if x != 'Unknown' else 'Unknown')
        
        # Strip whitespace
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        return df
    except FileNotFoundError:
        st.error("Netflix dataset file not found. Please upload the 'netflix_titles.csv' file.")
        return None

def main():
    # Main header
    st.markdown('<h1 class="main-header">🎬 Netflix Content Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_and_clean_data()
    
    if df is None:
        st.stop()
    
    # Sidebar for filters
    st.sidebar.title("🎯 Filters")
    
    # Content type filter
    content_types = st.sidebar.multiselect(
        "Select Content Type",
        options=df['type'].unique(),
        default=df['type'].unique()
    )
    
    # Year range filter
    year_range = st.sidebar.slider(
        "Select Release Year Range",
        min_value=int(df['release_year'].min()),
        max_value=int(df['release_year'].max()),
        value=(int(df['release_year'].min()), int(df['release_year'].max()))
    )
    
    # Filter data based on selections
    filtered_df = df[
        (df['type'].isin(content_types)) &
        (df['release_year'] >= year_range[0]) &
        (df['release_year'] <= year_range[1])
    ]
    
    # Key Metrics
    st.markdown('<h2 class="section-header">📊 Key Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Titles", len(filtered_df))
    
    with col2:
        movies_count = len(filtered_df[filtered_df['type'] == 'Movie'])
        st.metric("Movies", movies_count)
    
    with col3:
        tv_shows_count = len(filtered_df[filtered_df['type'] == 'TV Show'])
        st.metric("TV Shows", tv_shows_count)
    
    with col4:
        countries_count = filtered_df['primary_country'].nunique()
        st.metric("Countries", countries_count)
    
    # Content Type Distribution
    st.markdown('<h2 class="section-header">🎭 Content Type Distribution</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart using Plotly
        fig_pie = px.pie(
            filtered_df, 
            names='type', 
            title='Content Type Distribution',
            color_discrete_sequence=['#E50914', '#221F1F']
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart
        type_counts = filtered_df['type'].value_counts()
        fig_bar = px.bar(
            x=type_counts.index,
            y=type_counts.values,
            title='Content Count by Type',
            color=type_counts.index,
            color_discrete_sequence=['#E50914', '#221F1F']
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Genre Analysis
    st.markdown('<h2 class="section-header">🎪 Genre Analysis</h2>', unsafe_allow_html=True)
    
    top_genres = filtered_df['main_genre'].value_counts().head(15)
    
    fig_genres = px.bar(
        x=top_genres.values,
        y=top_genres.index,
        orientation='h',
        title='Top 15 Genres on Netflix',
        color=top_genres.values,
        color_continuous_scale='Reds'
    )
    fig_genres.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_genres, use_container_width=True)
    
    # Release Trend Analysis
    st.markdown('<h2 class="section-header">📈 Release Trends</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Content by release year
        yearly_counts = filtered_df['release_year'].value_counts().sort_index()
        fig_yearly = px.line(
            x=yearly_counts.index,
            y=yearly_counts.values,
            title='Content Release Trend by Year',
            markers=True
        )
        fig_yearly.update_traces(line_color='#E50914')
        st.plotly_chart(fig_yearly, use_container_width=True)
    
    with col2:
        # Content trend by type over years
        content_trend = filtered_df.groupby(['release_year', 'type']).size().reset_index(name='count')
        fig_trend = px.line(
            content_trend,
            x='release_year',
            y='count',
            color='type',
            title='Content Release Trend by Type',
            markers=True,
            color_discrete_sequence=['#E50914', '#221F1F']
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Rating Analysis
    st.markdown('<h2 class="section-header">⭐ Content Ratings</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution
        rating_counts = filtered_df['rating'].value_counts()
        fig_rating = px.bar(
            x=rating_counts.values,
            y=rating_counts.index,
            orientation='h',
            title='Content Rating Distribution',
            color=rating_counts.values,
            color_continuous_scale='Blues'
        )
        fig_rating.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_rating, use_container_width=True)
    
    with col2:
        # Rating by content type
        rating_type = filtered_df.groupby(['rating', 'type']).size().reset_index(name='count')
        fig_rating_type = px.bar(
            rating_type,
            x='count',
            y='rating',
            color='type',
            orientation='h',
            title='Rating Distribution by Content Type',
            color_discrete_sequence=['#E50914', '#221F1F']
        )
        st.plotly_chart(fig_rating_type, use_container_width=True)
    
    # Geographic Analysis
    st.markdown('<h2 class="section-header">🌍 Geographic Distribution</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top countries
        top_countries = filtered_df['primary_country'].value_counts().head(15)
        fig_countries = px.bar(
            x=top_countries.values,
            y=top_countries.index,
            orientation='h',
            title='Top 15 Countries by Content Count',
            color=top_countries.values,
            color_continuous_scale='Greens'
        )
        fig_countries.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_countries, use_container_width=True)
    
    with col2:
        # Country by content type
        country_type_counts = filtered_df.groupby(['primary_country', 'type']).size().reset_index(name='count')
        top_countries_list = filtered_df['primary_country'].value_counts().head(10).index
        country_type_filtered = country_type_counts[country_type_counts['primary_country'].isin(top_countries_list)]
        
        fig_country_type = px.bar(
            country_type_filtered,
            x='count',
            y='primary_country',
            color='type',
            orientation='h',
            title='Top 10 Countries by Content Type',
            color_discrete_sequence=['#E50914', '#221F1F']
        )
        st.plotly_chart(fig_country_type, use_container_width=True)
    
    # Duration Analysis
    st.markdown('<h2 class="section-header">⏱️ Duration Analysis</h2>', unsafe_allow_html=True)
    
    # Movie duration distribution
    movies_df = filtered_df[filtered_df['type'] == 'Movie']
    if len(movies_df) > 0:
        fig_duration = px.histogram(
            movies_df,
            x='duration_int',
            nbins=30,
            title='Movie Duration Distribution (Minutes)',
            color_discrete_sequence=['#E50914']
        )
        fig_duration.update_layout(
            xaxis_title='Duration (minutes)',
            yaxis_title='Number of Movies'
        )
        st.plotly_chart(fig_duration, use_container_width=True)
    
    # Top Directors and Cast
    st.markdown('<h2 class="section-header">🎬 Top Directors & Cast</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top directors
        director_counts = filtered_df[filtered_df['director'] != 'Unknown']['director'].str.split(', ').explode().value_counts().head(10)
        if len(director_counts) > 0:
            fig_directors = px.bar(
                x=director_counts.values,
                y=director_counts.index,
                orientation='h',
                title='Top 10 Directors',
                color=director_counts.values,
                color_continuous_scale='Purples'
            )
            fig_directors.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_directors, use_container_width=True)
    
    with col2:
        # Top cast members
        cast_counts = filtered_df[filtered_df['cast'] != 'Unknown']['cast'].str.split(', ').explode().value_counts().head(10)
        if len(cast_counts) > 0:
            fig_cast = px.bar(
                x=cast_counts.values,
                y=cast_counts.index,
                orientation='h',
                title='Top 10 Cast Members',
                color=cast_counts.values,
                color_continuous_scale='Oranges'
            )
            fig_cast.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_cast, use_container_width=True)
    
    # Content Addition Timeline
    st.markdown('<h2 class="section-header">📅 Content Addition Timeline</h2>', unsafe_allow_html=True)
    
    # Filter out null year_added values
    timeline_df = filtered_df.dropna(subset=['year_added', 'month_added'])
    
    if len(timeline_df) > 0:
        # Heatmap data
        heatmap_data = timeline_df.pivot_table(
            index='year_added',
            columns='month_added',
            values='show_id',
            aggfunc='count',
            fill_value=0
        )
        
        fig_heatmap = px.imshow(
            heatmap_data,
            title='Content Added to Netflix by Year and Month',
            color_continuous_scale='YlOrRd',
            aspect='auto'
        )
        fig_heatmap.update_layout(
            xaxis_title='Month',
            yaxis_title='Year',
            xaxis=dict(tickmode='linear', tick0=1, dtick=1)
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Data Table
    st.markdown('<h2 class="section-header">📋 Data Table</h2>', unsafe_allow_html=True)
    
    # Display options
    st.subheader("Filter and Search Data")
    
    # Search functionality
    search_term = st.text_input("Search in titles, directors, or cast:")
    
    if search_term:
        mask = (
            filtered_df['title'].str.contains(search_term, case=False, na=False) |
            filtered_df['director'].str.contains(search_term, case=False, na=False) |
            filtered_df['cast'].str.contains(search_term, case=False, na=False)
        )
        display_df = filtered_df[mask]
    else:
        display_df = filtered_df
    
    # Show data
    st.dataframe(
        display_df[['title', 'type', 'director', 'cast', 'country', 'release_year', 'rating', 'main_genre']].head(100),
        use_container_width=True
    )
    
    # Summary Statistics
    st.markdown('<h2 class="section-header">📈 Summary Statistics</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Overview")
        st.write(f"**Total Records:** {len(filtered_df):,}")
        st.write(f"**Movies:** {len(filtered_df[filtered_df['type'] == 'Movie']):,} ({len(filtered_df[filtered_df['type'] == 'Movie'])/len(filtered_df)*100:.1f}%)")
        st.write(f"**TV Shows:** {len(filtered_df[filtered_df['type'] == 'TV Show']):,} ({len(filtered_df[filtered_df['type'] == 'TV Show'])/len(filtered_df)*100:.1f}%)")
        st.write(f"**Unique Directors:** {filtered_df[filtered_df['director'] != 'Unknown']['director'].nunique():,}")
        st.write(f"**Unique Countries:** {filtered_df['primary_country'].nunique():,}")
        st.write(f"**Genres:** {filtered_df['main_genre'].nunique():,}")
    
    with col2:
        st.subheader("Release Year Statistics")
        st.write(f"**Earliest Release:** {filtered_df['release_year'].min()}")
        st.write(f"**Latest Release:** {filtered_df['release_year'].max()}")
        st.write(f"**Most Productive Year:** {filtered_df['release_year'].mode().iloc[0]} ({filtered_df['release_year'].value_counts().iloc[0]} titles)")
        
        if len(movies_df) > 0:
            st.write(f"**Average Movie Duration:** {movies_df['duration_int'].mean():.0f} minutes")
            st.write(f"**Shortest Movie:** {movies_df['duration_int'].min():.0f} minutes")
            st.write(f"**Longest Movie:** {movies_df['duration_int'].max():.0f} minutes")

if __name__ == "__main__":
    main()