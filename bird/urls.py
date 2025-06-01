from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login', views.login_page, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('signup', views.signup_page, name='signup'),
    path('explore/', views.explore, name='explore'),
    path('faq/', views.faq_view, name='faq')
]

if settings.DEBUG:  
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)