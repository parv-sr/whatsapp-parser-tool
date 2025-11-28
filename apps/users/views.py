from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required

from .forms import LoginForm

def login_view(request):
    if request.user.is_authenticated:
        return redirect("core:chat_view")

    form = LoginForm(request, data=request.POST or None)

    if request.method == "POST":
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect("core:chat_view")
        else:
            messages.error(request, "Invalid username or password")

    return render(request, "users/login.html", {"form": form})


def logout_view(request):
    logout(request)
    return redirect("users:login")
