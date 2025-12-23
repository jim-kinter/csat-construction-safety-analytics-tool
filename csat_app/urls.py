from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # Core pages
    path('', views.landing, name='landing'),
    path('risk-assessment/', views.RiskPredictionView.as_view(), name='risk_assessment'),
    path('analytics/', views.analytics_dashboard, name='analytics'),
    
    # Incidents
    path('incidents/', views.IncidentListView.as_view(), name='incident_list'),
    path('incident/<int:pk>/', views.IncidentDetailView.as_view(), name='incident_detail'),
    
    # Plans
    path('my-plans/', views.my_plans, name='my_plans'),
    path('load-plan/<int:plan_id>/', views.load_plan, name='load_plan'),
    path('save-plan/', views.save_plan, name='save_plan'),
    
    # Actions
    path('remove-activity/', views.remove_activity, name='remove_activity'),
    path('similar-incidents/<str:activity_name>/<str:equipment>/', 
         views.SimilarIncidentsView.as_view(), name='similar_incidents'),
    path('export-pdf/', views.export_pdf, name='export_pdf'),
    
    # Authentication
    path('logout/', views.logout_view, name='logout'),  # ← NOW INCLUDED

    # Daily Briefing
    path('generate-briefing/<int:plan_id>/', views.generate_briefing, name='generate_briefing'),
]