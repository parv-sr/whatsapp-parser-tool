from django import forms
from .widgets import MultiFileInput

class MultiTxtUploadForm(forms.Form):
    files = forms.FileField(
        widget=MultiFileInput(attrs={"multiple": True}),
        help_text="Upload one or more .txt WhatsApp export files.",
    )

    def clean_files(self):
        files = self.files.getlist("files") if hasattr(self.files, "getlist") else self.files
        # In Django forms, for multiple files, cleaned_data['files'] will be a list via request.FILES.getlist
        # We'll validate MIME and extension manually in the view because Django's FileField can't automatically handle multi-files easily.
        # (We'll implement extra checks in view.)
        return self.files
