from django.db import models

# Create your models here.
class Bird(models.Model):
    name = models.CharField(max_length=100)
    scientific_name = models.CharField(max_length=100)
    description = models.TextField()
    habitat = models.TextField()
    image = models.ImageField(upload_to='bird_images/', null=True, blank=True)
