from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('pose', views.pose, name='pose'),
    path('<music>', views.video, name='video'),
    path('pose/detectme', views.detectme, name='detectme')
] 
