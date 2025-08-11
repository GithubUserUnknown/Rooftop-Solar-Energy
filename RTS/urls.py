from django.urls import path
from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('get_weather/', views.get_weather, name='get_weather'),
    path('confirm_location_and_predict/', views.confirm_location_and_predict, name='confirm_location_and_predict'),
]
