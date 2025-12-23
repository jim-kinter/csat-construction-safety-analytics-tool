from django.utils.safestring import mark_safe
from django.views.generic import ListView
from django.contrib.auth import logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.shortcuts import render, get_object_or_404, redirect
from django.db.models import Q
from django.views import View
from django.utils import timezone
from django.http import HttpResponse
from django.template.loader import get_template
from django.conf import settings
from datetime import timedelta
from openai import OpenAI # type: ignore
import os

import json
import pandas as pd
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
from statsmodels.tsa.holtwinters import ExponentialSmoothing # type: ignore
from weasyprint import HTML # type: ignore
import joblib

from .models import Incident, SavedPlan, Location, User, IncidentType #, IncidentEmployee, Employee
from .forms import ActivityListForm

# ----------------------------------------------------------------------
# Incident Views
# ----------------------------------------------------------------------
class IncidentListView(ListView):
    model = Incident
    template_name = 'csat_app/incident_list.html'
    context_object_name = 'incidents'
    paginate_by = 50


class IncidentDetailView(View):
    def get(self, request, pk):
        incident = get_object_or_404(Incident, pk=pk)
        return render(request, 'csat_app/incident_detail.html', {'incident': incident})


# ----------------------------------------------------------------------
# Risk Assessment & Plan Management
# ----------------------------------------------------------------------
class RiskPredictionView(View):
    def get(self, request):
        activities = request.session.get('draft_activities', [])
        form = ActivityListForm()
        return render(request, 'csat_app/risk_assessment.html', {
            'form': form,
            'activities': activities
        })

    def post(self, request):
        form = ActivityListForm(request.POST)
        if form.is_valid():
            raw_text = form.cleaned_data['activities']
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

            draft = request.session.get('draft_activities', [])

            for line in lines:
                # New robust comma-separated parser
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 1:
                    continue  # skip empty/invalid lines

                name = parts[0]  # task description
                height = 0
                month = 'Unknown'
                state = 'Unknown'
                crew = 10  # default
                equipment = 'Unknown'

                if len(parts) > 1:
                    try:
                        height = int(parts[1])
                    except ValueError:
                        height = 0

                if len(parts) > 2:
                    month = parts[2]

                if len(parts) > 3:
                    state = parts[3]

                if len(parts) > 4:
                    try:
                        crew = int(parts[4])
                    except ValueError:
                        crew = 10

                if len(parts) > 5:
                    equipment = ', '.join(parts[5:])  # allow multi-word equipment

                prediction_data = self.run_prediction(name, equipment, height, month, state, crew)
                draft.append({
                    'name': name,
                    'equipment': equipment,
                    'height': height,
                    'month': month,
                    'state': state,
                    'crew': crew,
                    **prediction_data
                })

            request.session['draft_activities'] = draft
            return redirect('risk_assessment')

        return render(request, 'csat_app/risk_assessment.html', {
            'form': form,
            'activities': request.session.get('draft_activities', [])
        })

    def run_prediction(self, activity_name, equipment, height, month, state, crew):
        # Load model and encoders
        model = joblib.load('risk_model.joblib')
        le_weather = joblib.load('le_weather.joblib')
        le_equipment = joblib.load('le_equipment.joblib')
        le_severity = joblib.load('le_severity.joblib')

        # Dummy weather based on month/state
        weather = 'Hot' if month in ['June', 'July', 'August'] and state in ['TX', 'FL', 'AZ', 'GA', 'LA'] else 'Clear'

        try:
            weather_enc = le_weather.transform([weather])[0]
            equip_enc = le_equipment.transform([equipment])[0]
        except ValueError:
            weather_enc = 0
            equip_enc = 0

        # Month number for model
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        month_num = month_map.get(month.capitalize(), 6)  # default June

        # Safe input DataFrame (no chained assignment)
        input_df = pd.DataFrame().assign(
            weather_enc = [weather_enc],
            equip_enc = [equip_enc],
            month_enc = [month_num]
        )

        # Model prediction
        probas = model.predict_proba(input_df)[0]
        pred_class = model.predict(input_df)[0]
        severity = le_severity.inverse_transform([pred_class])[0]

        # Rule-based risk boost
        height_risk = min(height / 30.0, 1.0)
        crew_risk = min(max(crew - 5, 0) / 10.0, 1.0)
        month_risk = 0.5 if month in ['June', 'July', 'August'] else 0.0
        state_risk = 0.3 if state in ['TX', 'CA', 'FL', 'GA', 'LA'] else 0.0

        base_risk = max(probas)
        adjusted_risk = base_risk + (height_risk * 0.3) + (crew_risk * 0.2) + month_risk + state_risk
        risk_score = min(adjusted_risk, 1.0)

        # Recommendations
        recommendations = []
        if height > 20:
            recommendations.append("100% fall protection / tie-off required")
        if height > 50:
            recommendations.append("Personal fall arrest system mandatory")
        if month in ['June', 'July', 'August'] and state in ['TX', 'FL', 'AZ', 'GA', 'LA']:
            recommendations.append("Mandatory heat-illness prevention plan")
        if crew > 12:
            recommendations.append("Additional spotter/supervisor recommended")
        if 'crane' in equipment.lower():
            recommendations.append("Critical lift plan required")

        rec_text = ' • '.join(recommendations) if recommendations else 'Standard precautions sufficient.'

        # Final severity override
        if risk_score > 0.7:
            severity = 'High'
        elif risk_score > 0.4:
            severity = 'Medium'
        else:
            severity = 'Low'

        return {
            'risk_score': float(risk_score),  # ← Python float for JSON serialization
            'assessment': severity,
            'predicted_incident': f"Predicted {severity} severity incident",
            'recommendation': rec_text
        }
    
class SimilarIncidentsView(View):
    def get(self, request, activity_name, equipment):
        incidents = Incident.objects.filter(
            Q(equipment_involved__icontains=equipment) |
            Q(description__icontains=activity_name)
        ).order_by('-date')[:10]
        return render(request, 'csat_app/similar_incidents.html', {
            'activity_name': activity_name,
            'equipment': equipment,
            'incidents': incidents
        })


# ----------------------------------------------------------------------
# Plan Save / Load / List
# ----------------------------------------------------------------------
@login_required
def save_plan(request):
    if request.method == 'POST':
        plan_name = request.POST.get('plan_name', 'Untitled Plan')
        activities = request.session.get('draft_activities', [])

        if not activities:
            messages.error(request, "No activities to save.")
            return redirect('risk_assessment')

        # Save without requiring login — use session ID as identifier
        SavedPlan.objects.create(
            name=plan_name,
            created_by=None,  # no user required
            activities_json=activities
        )

        messages.success(request, f"Plan '{plan_name}' saved successfully!")
        request.session['draft_activities'] = []  # clear draft after save

    return redirect('my_plans')

@login_required
def my_plans(request):
    draft_activities = request.session.get('draft_activities', [])
    
    saved_plans = SavedPlan.objects.filter(created_by=request.user).order_by('-created_at')

    return render(request, 'csat_app/my_plans.html', {
        'activities': draft_activities,
        'saved_plans': saved_plans,
    })

@login_required
def load_plan(request, plan_id):
    plan = get_object_or_404(SavedPlan, id=plan_id, created_by=request.user)
    request.session['draft_activities'] = plan.activities_json
    messages.success(request, f"Plan '{plan.name}' loaded!")
    return redirect('risk_assessment')

# ----------------------------------------------------------------------
# Remove Activity
# ----------------------------------------------------------------------
def remove_activity(request):
    if request.method == "POST":
        activities = request.session.get('draft_activities', [])
        try:
            index = int(request.POST.get('index'))
            if 0 <= index < len(activities):
                removed = activities.pop(index)
                request.session['draft_activities'] = activities
                messages.success(request, f"Removed: {removed['name']} - {removed['equipment']}")
        except (ValueError, TypeError):
            messages.error(request, "Invalid activity index.")
    return redirect('risk_assessment')

# ----------------------------------------------------------------------
# Landing Page
# ----------------------------------------------------------------------
def landing(request):
    context = {
        'incident_count': Incident.objects.count(),
        'location_count': Location.objects.values('name').distinct().count(),
        'type_count': IncidentType.objects.count(),
    }
    return render(request, 'csat_app/landing.html', context)

# ----------------------------------------------------------------------
# Logout
# ----------------------------------------------------------------------
def logout_view(request):
    logout(request)
    return redirect('landing')

# ----------------------------------------------------------------------
# Export PDF
# ----------------------------------------------------------------------
@login_required
def export_pdf(request):
    activities = request.session.get('draft_activities', [])
    if not activities:
        messages.warning(request, "No activities to export.")
        return redirect('risk_assessment')

    template = get_template('csat_app/pdf_report.html')
    html = template.render({
        'activities': activities,
        'plan_name': request.GET.get('name', 'Current Plan'),
        'generated_at': timezone.now(),
    })

    pdf_file = HTML(string=html, base_url=request.build_absolute_uri()).write_pdf()

    response = HttpResponse(pdf_file, content_type='application/pdf')
    filename = f"CSAT_Safety_Report_{timezone.now().strftime('%Y%m%d_%H%M')}.pdf"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response

# ----------------------------------------------------------------------
# Analytics Dashboard
# ----------------------------------------------------------------------
def analytics_dashboard(request):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    plan_id = request.GET.get('plan')
    if plan_id and plan_id != 'all':
        try:
            user = User.objects.get(id=request.user.id)
            plan = SavedPlan.objects.get(id=plan_id, created_by=user)
            equipment_list = [a['equipment'] for a in plan.activities_json]
            incidents = Incident.objects.filter(equipment_involved__in=equipment_list)
            title_suffix = f" – Plan: {plan.name}"
        except (SavedPlan.DoesNotExist, User.DoesNotExist):
            incidents = Incident.objects.all()
            title_suffix = " – All Incidents"
    else:
        incidents = Incident.objects.all()
        title_suffix = " – All Incidents"

    df = pd.DataFrame(list(incidents.order_by('?').values('date', 'severity', 'weather', 'equipment_involved')[:50_000]))
    if df.empty:
        return render(request, 'csat_app/analytics.html', {'no_data': True})

    # === 1. Descriptive – Scatter Plot ===
    df_desc = df[df['date'].dt.year >= 1920].copy()
    df_desc['severity'] = df_desc['severity'].fillna('Unknown')

    color_map = {'Fatal': '#e74c3c', 'High': '#f39c12', 'Medium': '#2ecc71', 'Low': '#3498db', 'Unknown': '#95a5a6'}
    size_map = {'Fatal': 40, 'High': 20, 'Medium': 12, 'Low': 8, 'Unknown': 6}
    df_desc['marker_size'] = df_desc['severity'].map(size_map).fillna(6).astype(int)

    descriptive_scatter = px.scatter(
        df_desc, 
        x='date', 
        y='severity',
        size='marker_size', 
        color='severity',
        color_discrete_map=color_map,
        title=f"Descriptive – What Happened?{title_suffix}",
        hover_data=['equipment_involved'], 
        height=600
    )
    descriptive_scatter.update_layout(
        xaxis_title="Date",      
        yaxis_title="Severity",  
        yaxis=dict(categoryorder='array', categoryarray=['Low','Medium','High','Fatal','Unknown']),
        legend_title="Severity"
    )

    # === 2. Diagnostic – Weather vs Severity (Low/Medium/High only) ===
    df_diag = df[df['severity'].isin(['Low', 'Medium', 'High'])].copy()
    crosstab = pd.crosstab(df_diag['weather'], df_diag['severity'])
    crosstab = crosstab.reindex(columns=['Low','Medium','High'], fill_value=0)

    heatmap = go.Figure(data=go.Heatmap(
        z=crosstab.values, x=crosstab.columns, y=crosstab.index,
        colorscale='Reds', text=crosstab.values, texttemplate="%{text}"
    ))
    heatmap.update_layout(title="Diagnostic – Weather vs Severity (Low/Medium/High Only)")

    # === 3. Predictive – Risk Trend + Forecast (grouped + smoothed) ===
    df_trend = df[df['date'].dt.year >= 2023].copy()

    group_map = {
        'Electrical': ['panel', 'generator', 'wiring', 'switchgear', 'transformer', 'fuse', 'breaker'],
        'Mechanical': ['crane', 'forklift', 'vehicle', 'truck', 'dozer', 'excavator', 'barge', 'grader', 'plow'],
        'Construction Support': ['welder', 'grinder', 'torch', 'rigging', 'spotter', 'lockout device', 'barricade', 'h2s detector', 'saw', 'press', 'lathe', 'punch', 'tool'],
        'Construction Materials': ['tank', 'vessel', 'scaffold', 'steel', 'pipe', 'support', 'valve', 'compressor', 'fan', 'motor']
    }

    def categorize(equip):
        equip = str(equip).lower()
        for group, keywords in group_map.items():
            if any(k in equip for k in keywords):
                return group
        return 'Other'

    df_trend['category'] = df_trend['equipment_involved'].apply(categorize)
    monthly = df_trend.groupby([pd.Grouper(key='date', freq='ME'), 'category']).size().reset_index(name='count')
    pivot = monthly.pivot(index='date', columns='category', values='count').fillna(0).rolling(window=3, min_periods=1).mean()

    fig = go.Figure()
    colors = px.colors.qualitative.Set1

    for i, cat in enumerate(pivot.columns):
        series = pivot[cat]
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=cat,
                                line=dict(width=4, color=colors[i % len(colors)])))

        if len(series) >= 12:
            try:
                model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12).fit()
                forecast = model.forecast(12)
                forecast = forecast.clip(lower=0)  

                future = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=12, freq='ME')
                fig.add_trace(go.Scatter(x=future, y=forecast, mode='lines', name=f"{cat} (forecast)",
                                        line=dict(dash='dot', width=3, color=colors[i % len(colors)]), showlegend=False))
            except:
                pass

    fig.update_layout(title="Predictive – Equipment Risk Trend + 12-Month Forecast", height=650, xaxis_title="Date", yaxis_title='Incident Count', hovermode='x unified', legend_title='Equipment Category')

    # === 4. Tiny Equipment Group Legend (styled) ===
    group_definitions = {
        'Electrical': 'Electrical Panel, Generator, Wiring, Switchgear, Transformer, Fuse, Breaker',
        'Mechanical': 'Crane, Forklift, Vehicle, Truck, Dozer, Excavator, Barge, Grader, Plow',
        'Construction Support': 'Welder, Grinder, Torch, Rigging, Spotter, Lockout Device, Barricade, H2S Detector, Saw, Press, Lathe, Punch',
        'Construction Materials': 'Tank, Vessel, Scaffold, Steel, Pipe, Support, Valve, Compressor, Fan, Motor',
    }

    other_items = df_trend[df_trend['category'] == 'Other']['equipment_involved'].value_counts().head(5)
    other_text = ", ".join(other_items.index.tolist()) if not other_items.empty else "None"
    group_definitions['Other'] = f"Other equipment — most common: {other_text}"

    # Beautiful styled HTML table
    legend_html = '''
    <table style="font-size:0.9em; margin:20px auto; border-collapse:collapse; background:white; box-shadow:0 2px 8px rgba(0,0,0,0.1); border-radius:8px; overflow:hidden;">
        <thead>
            <tr style="background:#2c3e50; color:white;">
                <th style="padding:10px 20px; text-align:right;">Group</th>
                <th style="padding:10px 20px; text-align:left;">Contains</th>
            </tr>
        </thead>
        <tbody>
    '''

    for group, desc in group_definitions.items():
        legend_html += f'''
            <tr style="border-bottom:1px solid #eee;">
                <td style="padding:10px 20px; text-align:right; font-weight:600; color:#2c3e50;">{group}</td>
                <td style="padding:10px 20px; text-align:left; color:#444;">{desc}</td>
            </tr>
        '''

    legend_html += '''
        </tbody>
    </table>
    '''

    # === 4. Prescriptive – Mitigation Recommendations ===
    if plan_id:
        # PLAN-SPECIFIC: Show current loaded plan activities with full detail
        plan = get_object_or_404(SavedPlan, id=plan_id)
        activities = json.loads(plan.activities_json) if isinstance(plan.activities_json, str) else plan.activities_json

        rows = []
        for i, act in enumerate(activities, 1):
            equip = act.get('equipment', 'Unknown')
            weather = 'Hot' if act.get('month') in ['June','July','August'] and act.get('state') in ['TX','FL','AZ','GA','LA'] else 'Clear'

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a senior OSHA-certified construction safety engineer."},
                        {"role": "user", "content": f"""Activity: {act.get('name')}
    Height: {act.get('height', 0)} ft
    Month: {act.get('month', 'Unknown')}
    State: {act.get('state', 'Unknown')}
    Crew Size: {act.get('crew', 'Unknown')}
    Equipment: {equip}
    Weather: {weather}

    Provide exactly 3 specific, actionable mitigation recommendations using the hierarchy of controls (engineering → administrative → PPE)."""}
                    ],
                    max_tokens=180,
                    temperature=0.3
                )
                mitigation = response.choices[0].message.content.strip().replace('\n', '<br>')
            except:
                mitigation = "• Immediate job hazard analysis required<br>• Apply hierarchy of controls<br>• Consult site safety officer"

            # Build hover tooltip
            tooltip = (f"Height: {act.get('height', '—')} ft | "
                    f"Month: {act.get('month', '—')} | "
                    f"State: {act.get('state', '—')} | "
                    f"Crew: {act.get('crew', '—')} | "
                    f"Equipment: {equip}")

            risk_class = "high" if act.get('risk_score',0) > 0.8 else "medium" if act.get('risk_score',0) > 0.6 else "low"

            rows.append(f"""
            <tr data-risk="{risk_class}" style="background:{'#ffebee' if risk_class=='high' else '#fff3e0' if risk_class=='medium' else '#e8f5e8'}">
                <td>{i}</td>
                <td><span title="{tooltip}" style="cursor:help; border-bottom:1px dotted #3498db;">{act.get('name')}</span></td>
                <td>{act.get('height', '—')}</td>
                <td>{act.get('month', '—')}</td>
                <td>{act.get('state', '—')}</td>
                <td>{act.get('crew', '—')}</td>
                <td>{equip}</td>
                <td><strong>{act.get('risk_score', 0):.3f}</strong></td>
                <td>{mitigation}</td>
            </tr>
            """)

        prescriptive_table = mark_safe(f"""
            <h3>Prescriptive Analysis – Current Plan Mitigations</h3>
            <table class="table table-hover" style="font-size:0.94rem;">
                <thead class="table-dark">
                    <tr>
                        <th>#</th>
                        <th>Activity <small style="font-weight:normal">(hover for details)</small></th>
                        <th>Height (ft)</th>
                        <th>Month</th>
                        <th>State</th>
                        <th>Crew</th>
                        <th>Equipment</th>
                        <th>Risk Score</th>
                        <th>AI-Recommended Mitigations</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
            <p style="font-size:0.9em; color:#666; text-align:center;">
                Recommendations generated in real time using GPT-4o with full activity context
            </p>
        """)
    else:
        # ORIGINAL general prescriptive table (unchanged, safe fallback)
        high_risk = df[df['severity'].isin(['High', 'Fatal'])]
        top_equipment = high_risk['equipment_involved'].value_counts().head(8)
        rows = []
        for equip, count in top_equipment.items():
            weather = df[df['equipment_involved'] == equip]['weather'].mode()
            weather = weather[0] if not weather.empty else "Unknown"
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": "You are a construction safety engineer."},
                            {"role": "user", "content": f"Equipment: {equip}\nWeather: {weather}\nCaused {count} High/Fatal incidents\n\n3 bullet-point mitigations (engineering first)."}],
                    max_tokens=150, temperature=0.3)
                mitigation = response.choices[0].message.content.strip().replace('\n', '<br>')
            except:
                mitigation = "• Job hazard analysis<br>• Hierarchy of controls<br>• Consult safety officer"
            rows.append(f"<tr><td>{equip}</td><td>{count}</td><td>{weather}</td><td>{mitigation}</td></tr>")

        prescriptive_table = mark_safe(f"""
            <h3>Prescriptive – Top Equipment Mitigations (All History)</h3>
    <table class="table"><thead class="table-dark"><tr><th>Equipment</th><th>High/Fatal Incidents</th><th>Weather</th><th>AI Recommendations</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
        """)

    # Plan dropdown
    plans = []
    if request.user.is_authenticated:
        try:
            user = User.objects.get(id=request.user.id)
            plans = SavedPlan.objects.filter(created_by=user).order_by('-created_at')
        except:
            pass
    plan_choices = [('', 'All Incidents')] + [(str(p.id), p.name) for p in plans]

    charts = {
        'descriptive_scatter': descriptive_scatter.to_html(full_html=False, include_plotlyjs='cdn'),
        'diagnostic_heatmap': heatmap.to_html(full_html=False, include_plotlyjs='cdn'),
        'predictive_trend': fig.to_html(full_html=False, include_plotlyjs='cdn'),
        'legend_html': legend_html,
        'prescriptive_table': prescriptive_table,
        'plan_choices': plan_choices,
        'current_plan': plan_id or '',
    }

    return render(request, 'csat_app/analytics.html', charts)

#-------------------------------------------------
#
# Generate Briefing
#
#-------------------------------------------------

def generate_briefing(request, plan_id):
    plan = get_object_or_404(SavedPlan, id=plan_id)
    activities = plan.activities_json  # list of dicts

    # Get recent events (last 30 days)
    recent_incidents = Incident.objects.filter(date__gte=timezone.now() - timedelta(days=30))

    # Get trends (simple example – customize as needed)
    high_risk_equipment = recent_incidents.values('equipment_involved').annotate(count=models.Count('id')).order_by('-count')[:3]
    trends = [f"{eq['equipment_involved']} ({eq['count']} incidents)" for eq in high_risk_equipment]

    # Generate briefing with GPT-4o
    prompt = f"""
    Generate a simple daily safety briefing for a construction crew based on this plan:
    
    Plan Activities: {', '.join([act.get('name', 'Unknown') for act in activities])}
    
    Recent Events (last 30 days): {recent_incidents.count()} incidents reported.
    
    Trends: High-risk equipment includes {', '.join(trends)}.
    
    Briefing Structure:
    1. Weather Outlook (assume typical for site)
    2. Key Risks from Plan and Trends
    3. Mitigation Steps (engineering, administrative, PPE)
    4. Crew Reminders and Emergency Contacts
    
    Keep it concise (300-500 words), professional, and actionable.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600
        )
        briefing_text = response.choices[0].message.content.strip()
    except:
        briefing_text = "Briefing generation failed — use manual safety template."

    # Render PDF (using your existing export_pdf logic)
    template = get_template('csat_app/briefing_pdf.html') 
    html = template.render({'briefing_text': briefing_text, 'plan': plan})
    pdf = HTML(string=html).write_pdf()

    response = HttpResponse(pdf, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="daily_safety_briefing_{plan.name}.pdf"'
    return response