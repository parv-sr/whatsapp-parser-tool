from django.forms.widgets import FileInput

class MultiFileInput(FileInput):
    allow_multiple_selected = True
