from django.db import models
from django.contrib.auth.models import User

class User(models.Model):
    username = models.CharField(max_length=255)
    password_hash = models.CharField(max_length=255)
    role = models.CharField(max_length=50)

class Location(models.Model):
    name = models.CharField(max_length=255)
    address = models.TextField()
    site_manager = models.CharField(max_length=255, null=True)

#Removed from Scope. Retained for later implementation
#class Employee(models.Model):
#    name = models.CharField(max_length=255)
#    role = models.CharField(max_length=100, null=True)
#    experience_years = models.IntegerField(null=True)

class IncidentType(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(null=True)

class Incident(models.Model):
    date = models.DateTimeField()
    time = models.CharField(max_length=50, null=True)
    description = models.TextField(null=True)
    severity = models.CharField(max_length=50, null=True)
    weather = models.CharField(max_length=100, null=True)
    equipment_involved = models.CharField(max_length=255, null=True)
    cause = models.TextField(null=True)
    outcome = models.TextField(null=True)
    location = models.ForeignKey(Location, on_delete=models.SET_NULL, null=True)
    type = models.ForeignKey(IncidentType, on_delete=models.SET_NULL, null=True)
    data_source = models.CharField(max_length=50, default='unknown')

#Removed from Scope. Retained for later implementation
#class IncidentEmployee(models.Model):
#    incident = models.ForeignKey(Incident, on_delete=models.CASCADE)
#    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
#
#    class Meta:
#        unique_together = ('incident', 'employee')

class SavedPlan(models.Model):
    name = models.CharField(max_length=200, default="My Plan")
    created_by = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)
    activities_json = models.JSONField()  # stores the list of dicts

    def __str__(self):
        return f"{self.name} – {self.created_at.date()}"
    