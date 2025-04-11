from django.shortcuts import render

def index(request):
    """Render the homepage with the drawing canvas."""
    return render(request, 'index.html')
