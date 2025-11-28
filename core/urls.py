from django.urls import path
from apps.core import views
from apps.core.views import chat_view

app_name = "core"

urlpatterns = [
    path("chat/", chat_view, name="chat_view"),
    path("chat/query/", views.chat_query, name="chat_query"),
]
