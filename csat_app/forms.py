from django import forms

class ActivityListForm(forms.Form):
    activities = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 8, 'placeholder': 'One activity per line, [Task description, Height in feet, Month, State abbreviation, Crew size, Primary equipment (optional)] e.g.\nLift vessel - crane.'}),
        help_text="Enter each activity and its equipment separated by \" - \""
    )