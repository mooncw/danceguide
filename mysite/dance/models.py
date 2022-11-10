from django.db import models

class Dancemusic(models.Model):
    music_id = models.CharField(max_length=200, primary_key = True)
    music_name = models.CharField(max_length=200)
    music_path = models.CharField(max_length=200)
    def __str__(self):
        return self.music_name
