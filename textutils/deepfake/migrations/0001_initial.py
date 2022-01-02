# Generated by Django 3.2.10 on 2022-01-01 17:38

import deepfake.validator
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Video',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('videofile', models.FileField(null=True, upload_to='media/%y', validators=[deepfake.validator.file_size], verbose_name='')),
            ],
        ),
    ]