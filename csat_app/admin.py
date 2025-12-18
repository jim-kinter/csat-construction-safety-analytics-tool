from django.contrib import admin
from .models import SavedPlan, User, Location, IncidentType, Incident #,Employee, IncidentEmployee

admin.site.register(User)
admin.site.register(Location)
#admin.site.register(Employee)
admin.site.register(IncidentType)
admin.site.register(Incident)
#admin.site.register(IncidentEmployee)

@admin.register(SavedPlan)
class SavedPlanAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_by', 'created_at', 'activity_count')
    list_filter = ('created_at', 'created_by')
    search_fields = ('name', 'created_by__username')
    date_hierarchy = 'created_at'
    readonly_fields = ('created_at', 'activities_json')

    def activity_count(self, obj):
        return len(obj.activities_json)
    activity_count.short_description = "Activities"