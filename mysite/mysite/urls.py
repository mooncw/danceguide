from xml.etree.ElementInclude import include
from django.contrib import admin
from django.urls import path
from django.conf.urls import include
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('dance.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
