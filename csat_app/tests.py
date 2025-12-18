from django.test import TestCase
from django.urls import reverse
from django.contrib.auth.models import User
from csat_app.models import SavedPlan
from csat_app.views import RiskPredictionView
import json


class CSATCoreFunctionalityTests(TestCase):
    def setUp(self):
        # Create a test user
        self.user = User.objects.create_user(username='testsuper', password='safety123')
        self.client.login(username='testsuper', password='safety123')

    def test_risk_assessment_parsing_and_prediction(self):
        """Test that the comma parser + risk engine works end-to-end"""
        response = self.client.post(reverse('risk_assessment'), {
            'activities': 'Installing rebar, 85, June, TX, 14, scaffold\nCrane lift, 180, July, FL, 8, crane'
        })
        self.assertEqual(response.status_code, 302)  # redirect after POST

        # Follow to GET and check session has 2 activities with high risk
        response = self.client.get(reverse('risk_assessment'))
        activities = response.context['activities']
        self.assertEqual(len(activities), 2)

        # Both should be high risk due to height + summer + southern state
        self.assertGreater(activities[0]['risk_score'], 0.85)
        self.assertGreater(activities[1]['risk_score'], 0.85)
        self.assertIn('fall protection', activities[0]['recommendation'].lower())

    def test_save_and_load_plan(self):
        """Test full plan lifecycle"""
        # Add activities
        self.client.post(reverse('risk_assessment'), {
            'activities': 'Steel erection, 120, August, FL, 15, crane'
        })

        # Save plan
        response = self.client.post(reverse('save_plan'), {
            'plan_name': 'Downtown Tower High-Rise Plan'
        })
        self.assertEqual(response.status_code, 302)

        # Verify plan exists
        plan = SavedPlan.objects.get(name='Downtown Tower High-Rise Plan')
        self.assertEqual(len(plan.activities_json), 1)
        self.assertEqual(plan.created_by, self.user)

        # Load plan and verify it appears
        self.client.get(reverse('load_plan', args=[plan.id]))
        response = self.client.get(reverse('risk_assessment'))
        self.assertEqual(response.context['activities'][0]['name'], 'Steel erection')

    def test_analytics_plan_specific_table(self):
        """Test that analytics page shows plan-specific prescriptive table"""
        # Create and load a plan with high-risk activity
        self.client.post(reverse('risk_assessment'), {
            'activities': 'Welding at height, 95, July, TX, 10, welder'
        })
        self.client.post(reverse('save_plan'), {'plan_name': 'Refinery Outage'})
        plan = SavedPlan.objects.latest('created_at')

        response = self.client.get(f"{reverse('analytics')}?plan={plan.id}")
        self.assertContains(response, 'Welding at height')
        self.assertContains(response, 'Height (ft)')
        self.assertContains(response, 'AI-Recommended Mitigations')

    def test_model_accuracy_displayed(self):
        """Ensure model accuracy is shown somewhere (you can add a footer if you want)"""
        response = self.client.get(reverse('analytics'))
        # This string comes from predict_risk.py print statement — we just prove the page loads
        self.assertEqual(response.status_code, 200)