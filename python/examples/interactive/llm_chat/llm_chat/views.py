from django.http import HttpRequest, HttpResponse
from django.shortcuts import render


def index(request: HttpRequest) -> HttpResponse:
    """Render the homepage with the drawing canvas."""
    return render(request, 'index.html')
