from django import forms
from django.forms import ModelForm
from .models import Video

class VideoForm(forms.ModelForm):
    class Meta:
        model= Video
        fields= ["videofile"]

