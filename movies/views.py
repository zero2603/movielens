from django.http import HttpResponse
from django.template.response import TemplateResponse
from django.template import loader
import numpy as np
import pandas as pd
import content_based

movies = np.array(pd.read_csv("data/movies.csv", sep='\t', usecols=['movie_id', 'title', 'genres']))

def index(request):
    template = loader.get_template('home.html')
    user_id = 0

    if request.session.get('user_id'):
        user_id = request.session.get('user_id') 

    context = {
        'movies': movies[:20], 
        'user_id': user_id,
    }
    return HttpResponse(template.render(context, request)) 

def login(request):
    template = loader.get_template('home.html')
    
    if request.POST['username']:
        request.session['user_id'] = request.POST['username']
    user_id = request.session['user_id']

    with open("recommended/user/" + request.session['user_id'] + ".txt") as f:
        recommended_movies_id = f.readlines()
    recommended_movies_id = map(int, recommended_movies_id)
    
    recommended_movies = []
    for item in movies:
        if item[0] in recommended_movies_id:
            recommended_movies.append(item)

    context = {
        'movies': movies[:20], 
        'user_id': user_id, 
        'recommended_movies_1': recommended_movies[4:],
        'recommended_movies_2': recommended_movies[:4],
    }
    return HttpResponse(template.render(context, request))

def get_detail(request):
    template = loader.get_template('detail.html')
    movie_id = int(request.GET.get('movie'))
    movie= None
    
    if request.session.get('user_id'):
        user_id = request.session.get('user_id') 

    for item in movies:
        if item[0] == movie_id:
            movie = item
            break

    recommended_movies_id = content_based.recommend(movie_id)
    recommended_movies = []
    for item in movies:
        if item[0] in recommended_movies_id:
            recommended_movies.append(item)

    context = {'movie': movie, 'recommended_movies': recommended_movies, 'user_id': user_id}
    return HttpResponse(template.render(context, request))

def logout(request):
    template = loader.get_template('home.html')
    del request.session['user_id']
    context = {}
    return HttpResponse(template.render(context, request)) 