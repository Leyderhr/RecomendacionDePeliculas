import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS as stopwords_wc

# Cargar el DataFrame original
df = pd.read_csv("IMDB_Top250Movies.csv")

# Formatear los años para eliminar los separadores de miles
df['year'] = df['year'].apply(lambda x: str(x).replace(',', ''))

# Divide la columna de género en columnas separadas (usando get_dummies para la codificación one-hot)
df_genre = df['genre'].str.get_dummies(sep=',')

# Combina el DataFrame original con el DataFrame de género
df = pd.concat([df, df_genre], axis=1)

# Calcular la matriz de similitud de coseno
features = df_genre.columns
similarity_matrix = cosine_similarity(df[features], df[features])


# Diccionario para almacenar las preferencias del usuario
prefencias_usuario = {
    'likes':[],
    'dislikes': []
}

# Función para obtener recomendaciones
def obtener_recomendaciones(pelicula_referencia, similarity_matrix, df, n=10):
    indice_referencia = df[df['name'] == pelicula_referencia].index[0]
    similitud_pelicula_referencia = similarity_matrix[indice_referencia]
    puntuaciones_similitud = list(enumerate(similitud_pelicula_referencia))
    puntuaciones_similitud = sorted(puntuaciones_similitud, key=lambda x: x[1], reverse=True)
    recomendaciones = puntuaciones_similitud[1:n+1]
    nombres_recomendados = [df.iloc[i]['name'] for i, _ in recomendaciones]
    return nombres_recomendados


# Inicializar el estado de la sesión para guardar las preferencias del usuario
if 'likes' not in st.session_state:
    st.session_state['likes'] = []

if 'dislikes' not in st.session_state:
    st.session_state['dislikes'] = []
 
   
# Función para actualizar el perfil del usuario
def actualizar_perfil_usuario(df, likes, dislikes):
    generos_like = df[df['name'].isin(likes)][df_genre.columns].sum()
    generos_dislike = df[df['name'].isin(dislikes)][df_genre.columns].sum()
    perfil_usuario = generos_like - generos_dislike
    return perfil_usuario


# Función para recomendar películas basadas en el perfil del usuario
def recomendar_peliculas(perfil_usuario, df):
    similitud_perfil = cosine_similarity([perfil_usuario], df[df_genre.columns])
    puntuaciones_similitud = list(enumerate(similitud_perfil[0]))
    puntuaciones_similitud = sorted(puntuaciones_similitud, key=lambda x: x[1], reverse=True)
    recomendaciones = puntuaciones_similitud[:10]
    nombres_recomendados = [df.iloc[i]['name'] for i, _ in recomendaciones]
    return nombres_recomendados
    


# Función para recomendar películas basadas en preferencias
def recomndar_por_preferencias(preferencias_usuario, df, similarity_matrix):
    generos_like = df[df['name'].isin(prefencias_usuario['likes'])][df_genre.columns].sum()
    generos_dislike = df[df['name'].isin(preferencias_usuario['dislikes'])][df_genre.columns].sum()
    perfil_usuario = generos_like - generos_dislike
    
    
    similitud_perfil = cosine_similarity([perfil_usuario], df[df_genre.columns])
    puntuaciones_similitud = list(enumerate(similitud_perfil[0]))
    puntuaciones_similitud = sorted(puntuaciones_similitud, key=lambda x: x[1], reverse=True)
    
    
    recomendaciones = puntuaciones_similitud[:10]
    nombres_recomendados = [df.iloc[i]['name'] for i, _ in recomendaciones]
    return nombres_recomendados


# Función para EDA
def eda_section():
    st.header("Análisis Exploratorio de Datos (EDA)")

    # Top del rating según el género de las películas
    st.subheader("Top del Rating según el Género")
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    top_rates = df.groupby('genre')['rating'].mean().sort_values(ascending=False).head(10)
    top_rates_df = pd.DataFrame({'Género': top_rates.index, 'Calificación': top_rates.values})
    top_rates_df['Calificación'] = top_rates_df['Calificación'].apply(lambda x: f'{x:.1f}')
    st.table(top_rates_df)

    # Gráfico de barras para representar las 10 películas con las calificaciones más altas, clasificadas por género
    st.subheader("Top 10 Películas por Género")
    top_rates = df.nlargest(10, 'rating')
    fig = go.Figure(data=[go.Bar(x=top_rates['genre'], y=top_rates['rating'], marker={'color': '#00FA9A'})])
    st.plotly_chart(fig)

     # Gráfico de barras para representar las 10 películas con las calificaciones más altas
    st.subheader("Top 10 Películas por Calificación (Rating)")
    top_movies = df.nlargest(10, 'rating')
    fig = go.Figure(data=[go.Bar(x=top_movies['name'], y=top_movies['rating'], marker={'color': '#3CB371'})])
    st.plotly_chart(fig)

    # Gráfico de barras para representar los 10 directores principales según las calificaciones más altas
    st.subheader("Top 10 Películas por Directores")
    top_movies_d = df.nlargest(10, 'rating')
    fig_directors = go.Figure(data=[go.Bar(x=top_movies_d['directors'], y=top_movies_d['rating'], marker={'color': '#2E8B57'})])
    st.plotly_chart(fig_directors)

    # Histograma que representa la distribución de los ratings en el conjunto de datos
    st.subheader("Distribución de Ratings")
    plt.hist(df['rating'], bins=20, color='green')
    plt.xlabel('Rating')
    plt.ylabel('Frecuencia')
    st.pyplot(plt.gcf()) 

    # Sunburst que representa cómo se distribuyen las calificaciones de las películas en función de su género y año de lanzamiento
    st.subheader("Calificaciones de Películas por Género y Año")
    agrupar_df = df.groupby(['genre', 'year', 'rating']).size().reset_index(name='count')
    fig_sunburst = px.sunburst(agrupar_df, path=['genre', 'year', 'rating'], values='count')
    fig_sunburst.update_layout( height=1100, width=1100)
    st.plotly_chart(fig_sunburst)

    # Gráfico de línea que visualiza cómo la calificación promedio de las películas ha cambiado a lo largo de los años
    st.subheader("Calificación Promedio a lo Largo del Tiempo")
    agrupar_df = df.groupby('year')['rating'].mean().reset_index()
    fig_line = px.line(agrupar_df, x='year', y='rating', title='Calificación Promedio a lo Largo del Tiempo', color_discrete_sequence=['#00FF00'])
    fig_line = px.line(agrupar_df, x='year', y='rating', color_discrete_sequence=['#00FF00'])
    fig_line.update_layout(xaxis_title='Año', yaxis_title='Calificación Promedio')
    st.plotly_chart(fig_line)

    # Pie Chart para analizar cómo distribuyen las calificaciones promedio de las películas en diferentes certificados
    st.subheader("Calificaciones Promedio por Certificado")
    avg_ratings = df.groupby('certificate')['rating'].mean().reset_index()
    fig_pie = px.pie(avg_ratings, names='certificate', values='rating')
    st.plotly_chart(fig_pie)

    # Gráfico de tabla que proporciona una visualización clara de las 10 películas más antiguas
    st.subheader("Películas Más Antiguas")
    peliculas_antiguas = df.sort_values(by='year').head(10)
    fig_table_antiguas = go.Figure(data=[
        go.Table(
            header=dict(values=['Nombre de la Película', 'Año de Lanzamiento'], fill_color='green'),
            cells=dict(values=[peliculas_antiguas['name'], peliculas_antiguas['year']])
        )
    ])
    st.plotly_chart(fig_table_antiguas)

    # Gráfico de tabla que proporciona una visualización clara de las 10 películas más recientes
    st.subheader("Películas Más Recientes")
    peliculas_recientes = df.sort_values(by='year', ascending=False).head(10)
    fig_table_recientes = go.Figure(data=[go.Table(header=dict(values=['Nombre de la Película', 'Año de Lazamiento'], fill_color='green'),
                                                    cells=dict(values=[peliculas_recientes['name'], peliculas_recientes['year']]))
                                            ])
    st.plotly_chart(fig_table_recientes)


    # Nube de palabras para visualizar de manera efectiva los géneros de películas más frecuentes en el conjunto de datos
    st.subheader("Nube de Palabras de Géneros de Películas")
    all_genres = " ".join(token for token in df["genre"])
    wordcloud = WordCloud(stopwords=stopwords_wc,
                        max_words=1500,
                        max_font_size=350, random_state=42,
                        width=2000, height=1000,
                        colormap="viridis",
                        contour_color='steelblue')
    wordcloud.generate(all_genres)
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt.gcf())

    # Verificación de la cantidad de películas para cada certificado en el DataFrame
    st.subheader("Cantidad de Películas para cada Certificado")
    certificate_counts = df['certificate'].value_counts()
    st.table(certificate_counts)



# Función para sistema de recomendación
def recommend_section():
    st.header("Sistema de Recomendación")

    # Crear un menú desplegable con la lista de películas para el sistema de recomendación
    peliculas = df.apply(lambda x: f"{x['name']} ({x['year']}) - {x['genre']}", axis=1)
    seleccion = st.selectbox('Selecciona una película de referencia', peliculas)
    pelicula_referencia = seleccion.split('(')[0].strip()
    col1, col2, col3 = st.columns([1, 1, 4])
    
    # Botón para indicar que le gusta la película
    with col1:
        if st.button("Like"):
            if pelicula_referencia not in st.session_state['likes']:
                st.session_state['likes'].append(pelicula_referencia)
            if pelicula_referencia in st.session_state['dislikes']:
                st.session_state['dislikes'].remove(pelicula_referencia)

        
    # Botón para indicar que le gusta la película 
    with col2:
        if st.button("Dislike"):
            if pelicula_referencia not in st.session_state['dislikes']:
                st.session_state['dislikes'].append(pelicula_referencia)
            if pelicula_referencia in st.session_state['likes']:
                st.session_state['likes'].remove(pelicula_referencia)
        
    
    
    perfil_usuario = actualizar_perfil_usuario(df, st.session_state['likes'], st.session_state['dislikes'])
    recomendaciones = recomendar_peliculas(perfil_usuario, df)



    # Generear recomendaciones basadas en preferencias
    # recomendaciones = recomndar_por_preferencias(prefencias_usuario, df, similarity_matrix)
    if(recomendaciones):
        st.subheader("Recomendaciones basadas en tus preferencias:")
        table_data = []
        
        # Excluir la película de referencia de las recomendaciones
        recomendaciones = [r for r in recomendaciones if r != pelicula_referencia]
        
        for recomendacion in recomendaciones:
            if recomendacion != pelicula_referencia:
                info_adicional = df[df['name'] == recomendacion][['year', 'run_time', 'genre']].values

                # Asegurarse de que info_adicional es un array unidimensional
                if len(info_adicional) > 0:
                    info_adicional = info_adicional[0]

                    # Añadir datos a la tabla
                    table_data.append({
                        'Name': recomendacion,
                        'Year': info_adicional[0],
                        'Runtime': info_adicional[1],
                        'Genre': info_adicional[2]
                    })
        
        if table_data:
        # Intentar crear el DataFrame
            df_table = pd.DataFrame(table_data)
            
            # Mostrar la tabla
            st.table(df_table[['Name', 'Year', 'Runtime', 'Genre']])
        
    else:
        st.write("No se encontraron recomendaciones basadas en tus preferencias") 
           
    
    # Mostrar tablas de películas que le gustan y no le gustan
    st.subheader("Películas que te gustan")
    if st.session_state['likes']:
        peliculas_like = df[df['name'].isin(st.session_state['likes'])]
        st.table(peliculas_like[['name', 'year', 'genre']])
    else:
        st.write("Aún no has indicado ninguna película que te guste.") 
        
        
    st.subheader("Películas que no te gustan")
    if st.session_state['dislikes']:
        peliculas_like = df[df['name'].isin(st.session_state['dislikes'])]
        st.table(peliculas_like[['name', 'year', 'genre']])
    else:
        st.write("Aún no has indicado ninguna película que te guste.")   
            

# Sidebar con opciones
sidebar_option = st.sidebar.radio("Selecciona una opción", ["Inicio", "Análisis Exploratorio de Datos", "Sistema de Recomendación"])

# Contenido principal
if sidebar_option == "Inicio":
    st.title("Bienvenido a FilmFinder!")
    st.markdown("Explora películas y descubre nuevas recomendaciones.")

elif sidebar_option == "Análisis Exploratorio de Datos":
    eda_section()

elif sidebar_option == "Sistema de Recomendación":
    recommend_section()
