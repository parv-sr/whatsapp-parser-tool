from django.urls import path
from . import views
from .views import chat_view

app_name = "core"

urlpatterns = [
    path("chat/", chat_view, name="chat_view"),
    path("chat/query/", views.chat_query, name="chat_query"),
    path("chat/stream/", views.chat_stream, name="chat_stream"),
    path("chat/source/<int:pk>/", views.get_listing_source, name="get_listing_source"),
]
