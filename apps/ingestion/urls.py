from django.urls import path
from . import views

app_name = "ingestion"

urlpatterns = [
    path('', views.upload_redirect, name='upload_redirect'),
    path("upload/", views.upload_files, name="upload_files"),
    path("uploads/", views.uploads_list, name="uploads_list"),
    path('upload/success/', views.upload_success, name='upload_success'),
    path("progress/", views.progress_status, name="progress_status"),
]
