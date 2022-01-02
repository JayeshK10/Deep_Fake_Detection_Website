from django.db import models
from .validator import file_size

# Create your models here.
class Video(models.Model):
    videofile= models.FileField(upload_to='media/%y', validators = [file_size] , null=True, verbose_name="")

    def __str__(self):
        return str(self.videofile)